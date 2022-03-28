# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Evaluating trained retrieval model.

Usage:
python eval_mhop_retrieval.py ${EVAL_DATA} ${CORPUS_VECTOR_PATH} ${CORPUS_DICT} ${MODEL_CHECKPOINT} \
     --batch-size 50 \
     --beam-size-1 20 \
     --beam-size-2 5 \
     --topk 20 \
     --shared-encoder \
     --gpu \
     --save-path ${PATH_TO_SAVE_RETRIEVAL}

"""
import argparse
import collections
import json
import logging
import os
from os import path
import time
from datetime import datetime

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from mdr.retrieval.models.mhop_retriever import RobertaRetriever
from mdr.retrieval.utils.basic_tokenizer import SimpleTokenizer
from mdr.retrieval.utils.utils import (load_saved, move_to_cuda, para_has_answer)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

def convert_hnsw_query(query_vectors):
    aux_dim = np.zeros(len(query_vectors), dtype='float32')
    query_nhsw_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
    return query_nhsw_vectors

def get_candidate_sentences(cand, title2sents_map):
    new_text = cand['text']
    if type(new_text) == list:
        return cand
    if cand['title'] in title2sents_map:
        new_text = title2sents_map[cand['title']]
        cand['text'] = new_text
    else:
        new_text = title2sents_map[f"PERSONAL_{cand['title']}"]
        cand['text'] = new_text
    return cand

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_data', type=str, default=None)
    parser.add_argument('indexpath', type=str, default=None)
    parser.add_argument('corpus_dict', type=str, default=None)
    parser.add_argument('model_path', type=str, default=None)
    parser.add_argument('--topk', type=int, default=2, help="topk paths")
    parser.add_argument('--num-workers', type=int, default=10)
    parser.add_argument('--max-q-len', type=int, default=70)
    parser.add_argument('--max-c-len', type=int, default=300)
    parser.add_argument('--max-q-sp-len', type=int, default=350)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--beam-size', type=int, default=5)
    parser.add_argument('--model-name', type=str, default='roberta-base')
    parser.add_argument('--gpu', action="store_true")
    parser.add_argument('--gpu_num', type=int, default=6, help="Gpus for the index.")
    parser.add_argument('--save-index', action="store_true")
    parser.add_argument('--only-eval-ans', action="store_true")
    parser.add_argument('--shared-encoder', action="store_true")
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--indices-path", type=str, default="", help="data points to debug")
    parser.add_argument("--metrics-path", type=str, default="")
    parser.add_argument("--stop-drop", default=0, type=float)
    parser.add_argument('--hnsw', action="store_true")
    parser.add_argument('--only_hop_1', default=False, help="whether to only run the first hop of retrieval")
    parser.add_argument('--name', default=False, help="experiment name")

    # multi-index new args
    parser.add_argument('--indexpath_alt', type=str, default=None, help="private dataset index")
    parser.add_argument('--corpus_dict_alt', type=str, default=None, help="private dataset corpus")
    parser.add_argument('--doc_privacy', type=int, default=0, help="whether to restrict local --> global path of retrieval due to document privacy")
    parser.add_argument('--min_retrieved_by_domain', type=int, default=0, help="enforces that this min amount of passages are selected from both domains after ranking top k")
    parser.add_argument('--query_privacy', type=int, default=0, help="whether to only ask a query locally due to query privacy")
    parser.add_argument("--retrieval_mode", 
        type=str, 
        default=None,
        choices=["4combo_separaterank", "4combo_overallrank", "fullindex", "2combo_oracledomain"],
        help="Which method to use to retrieve documents; either oracle, or rank all four versions, or use one full index"
    )  
    parser.add_argument("--title2sents_map",
        type=str,
        default="/mnt/disks/scratch1/concurrentqa/datasets/concurrentqa/corpora/title2sent_map.json"
    )

    args = parser.parse_args()

    print(f"Run: {args.name}")

    indices = []
    if args.indices_path:
        with open(args.indices_path) as f:
            for line in f:
                indices.append(line.strip("\n"))

    if not os.path.exists(os.path.dirname(args.save_path)):
        print(os.path.dirname(args.save_path))
        os.makedirs(os.path.dirname(args.save_path))
    else:
        out_dirname = os.path.dirname(args.save_path)
        now = datetime.now()
        datetime_str = now.strftime("%d/%m/%Y %H:%M:%S")
        with open(f"{out_dirname}/retrieval_configs.json", "w") as f:
            f.write(f"DATE: {datetime_str}\n")
            json.dump(vars(args), f)   

    if not args.save_path:
        assert 0, "empty save path"
    if not args.metrics_path:
        assert 0, "empty metrics path"
         
    
    # load the qa pairs
    logger.info("Loading data...")
    ds_items = [json.loads(_) for _ in open(args.raw_data).readlines()]


    logger.info("Loading title2sents map...")
    with open(args.title2sents_map) as f:
        title2sents_map = json.load(f)
    logger.info(f"Size: {len(title2sents_map)}")

    # filter
    if args.only_eval_ans:
        ds_items = [_ for _ in ds_items if _["answer"][0] not in ["yes", "no"]]

    # load the model 
    logger.info("Loading trained model...")
    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = RobertaRetriever(bert_config, args)
    model = load_saved(model, args.model_path, exact=False)
    simple_tokenizer = SimpleTokenizer()

    cuda = torch.device('cuda')
    model.to(cuda)
    from apex import amp
    model = amp.initialize(model, opt_level='O1')
    model.eval()

    # load the encoded passages and build the indices 
    logger.info("Building index...")
    d = 768
    xb = np.load(args.indexpath).astype('float32')

    if args.indexpath_alt:
        logger.info("Building alternate index...")
        xb_alt = np.load(args.indexpath_alt).astype('float32')

    if args.hnsw:
        if path.exists("data/hotpot_index/wiki_index_hnsw.index"):
            index = faiss.read_index("index/wiki_index_hnsw.index")
        else:
            index = faiss.IndexHNSWFlat(d + 1, 512)
            index.hnsw.efSearch = 128
            index.hnsw.efConstruction = 200
            phi = 0
            for i, vector in enumerate(xb):
                norms = (vector ** 2).sum()
                phi = max(phi, norms)
            logger.info('HNSWF DotProduct -> L2 space phi={}'.format(phi))

            data = xb
            buffer_size = 50000
            n = len(data)
            print(n)
            for i in tqdm(range(0, n, buffer_size)):
                vectors = [np.reshape(t, (1, -1)) for t in data[i:i + buffer_size]]
                norms = [(doc_vector ** 2).sum() for doc_vector in vectors]
                aux_dims = [np.sqrt(phi - norm) for norm in norms]
                hnsw_vectors = [np.hstack((doc_vector, aux_dims[idx].reshape(-1, 1))) for idx, doc_vector in enumerate(vectors)]
                hnsw_vectors = np.concatenate(hnsw_vectors, axis=0)
                index.add(hnsw_vectors)
    else:
        index = faiss.IndexFlatIP(d)
        index.add(xb)
        if args.gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, args.gpu_num, index)

        if args.indexpath_alt:
            index_alt = faiss.IndexFlatIP(d)
            index_alt.add(xb_alt)
            if args.gpu:
                res_alt = faiss.StandardGpuResources()
                index_alt = faiss.index_cpu_to_gpu(res_alt, args.gpu_num, index_alt)

    if args.save_index:
        faiss.write_index(index, "data/hotpot_index/wiki_index_hnsw_roberta")
    
    # load the passages in raw format
    logger.info(f"Loading corpus...")
    id2doc = json.load(open(args.corpus_dict))
    if isinstance(id2doc["0"], list):
        id2doc = {k: {"title":v[0], "text": v[1]} for k, v in id2doc.items()}
        
    #title2text = {v[0]:v[1] for v in id2doc.values()}
    logger.info(f"Corpus size {len(id2doc)}")

    if args.corpus_dict_alt:
        logger.info(f"Loading corpus alt...")
        id2doc_alt = json.load(open(args.corpus_dict_alt))
        if isinstance(id2doc_alt["0"], list):
            id2doc_alt = {k: {"title":v[0], "text": v[1]} for k, v in id2doc_alt.items()}
        logger.info(f"Corpus size {len(id2doc_alt)}")
    elif args.retrieval_mode != "fullindex":
        id2doc_alt = None

    logger.info("Encoding questions and searching")
    questions = [_["question"][:-1] if _["question"].endswith("?") else _["question"] for _ in ds_items]
    ds_indices = [_["_id"][:-1] if _["_id"].endswith("?") else _["_id"] for _ in ds_items]

    has_domains = 0
    if "domain" in list(ds_items[0].keys()):
        has_domains = 1
        if "CQA" not in args.raw_data:
            domains = [_["domain"] for _ in ds_items]
        else:
            domains = []
            for d_ex in ds_items:
                sp_ex = d_ex["sp"]
                domain_ex = []
                sptitle0 = sp_ex[0]['title']
                if "e" in sptitle0 and "_p" in sptitle0:
                    domain_ex.append(1)
                else:
                    domain_ex.append(0)

                sptitle1 = sp_ex[1]['title']
                if "e" in sptitle1 and "_p" in sptitle1:
                    domain_ex.append(1)
                else:
                    domain_ex.append(0)
                domains.append(domain_ex)
        

    metrics = []
    retrieval_outputs = []
    hop1relevance_domain0 = collections.defaultdict(dict)
    hop1relevance_domain1 = collections.defaultdict(dict)
    hop2relevance_domain1_domain1 = collections.defaultdict(dict)
    hop2relevance_domain1_domain2 = collections.defaultdict(dict)
    hop2relevance_domain2_domain2 = collections.defaultdict(dict)
    hop2relevance_domain2_domain1 = collections.defaultdict(dict)
    num_questions = len(questions) 
    for b_start in tqdm(range(0, num_questions, args.batch_size)):
        with torch.no_grad():
            batch_q = questions[b_start:b_start + args.batch_size] 
            batch_idxs = ds_indices[b_start:b_start + args.batch_size] 
            batch_ann = ds_items[b_start:b_start + args.batch_size]
            if has_domains:
                batch_domains = domains[b_start:b_start + args.batch_size]
            bsize = len(batch_q)

            batch_q_encodes = tokenizer.batch_encode_plus(batch_q, max_length=args.max_q_len, pad_to_max_length=True, return_tensors="pt")
            batch_q_encodes = move_to_cuda(dict(batch_q_encodes))
            q_embeds = model.encode_q(batch_q_encodes["input_ids"], batch_q_encodes["attention_mask"], batch_q_encodes.get("token_type_ids", None))
            q_embeds_numpy = q_embeds.cpu().contiguous().numpy()
            if args.hnsw:
                q_embeds_numpy = convert_hnsw_query(q_embeds_numpy)

            # HOP 1
            # get the nearest neighbors for each embedded question    
            D, I = index.search(q_embeds_numpy, args.beam_size) 
            if args.indexpath_alt:
                D_alt, I_alt = index_alt.search(q_embeds_numpy, args.beam_size) 
            elif args.retrieval_mode != "fullindex":
                D_alt, I_alt = index.search(q_embeds_numpy, args.beam_size) 
            else:
                D_alt, I_alt = None, None

            # rank all the documents collected from the first hop on both indices
            # we don't do this for query privacy since there we will just use D_alt
            if D_alt is not None and not args.query_privacy:
                D_tops, D_alt_tops = [], []
                I_tops, I_alt_tops = [], []
                for b_idx in range(bsize):
                    D_arr = D[b_idx]
                    D_alt_arr = D_alt[b_idx]
                    I_arr = I[b_idx]
                    I_alt_arr = I_alt[b_idx]
                    D_top, D_alt_top = [], []
                    I_top, I_alt_top = [], []
                    pos, pos_alt = 0, 0

                    if args.min_retrieved_by_domain > 0:
                        assert len(D_arr) > args.min_retrieved_by_domain, print("The min retrieved by domain is larger than the number of passages in the domain")
                        assert len(D_alt_arr) > args.min_retrieved_by_domain, print("The min retrieved by alt domain is larger than the number of passages in the domain")
                        while pos < args.min_retrieved_by_domain:
                            D_top.append(D_arr[pos])
                            I_top.append(I_arr[pos])
                            pos += 1
                        while pos_alt < args.min_retrieved_by_domain:
                            D_alt_top.append(D_alt_arr[pos_alt])
                            I_alt_top.append(I_alt_arr[pos_alt])
                            pos_alt += 1

                    # merge sort
                    while (len(D_top) + len(D_alt_top)) < args.beam_size:
                        if pos < len(D_arr) and D_arr[pos] >= D_alt_arr[pos_alt]:
                            D_top.append(D_arr[pos])
                            I_top.append(I_arr[pos])
                            pos += 1
                        elif pos_alt < len(D_alt_arr) and D_arr[pos] < D_alt_arr[pos_alt]:
                            D_alt_top.append(D_alt_arr[pos_alt])
                            I_alt_top.append(I_alt_arr[pos_alt])
                            pos_alt += 1

                    # fill remaining slots with -inf
                    while len(D_top) < args.beam_size:
                        D_top.append(float("-inf"))
                        I_top.append(-1)
                    while len(D_alt_top) < args.beam_size:
                        D_alt_top.append(float("-inf"))
                        I_alt_top.append(-1)
                    D_tops.append(D_top)
                    D_alt_tops.append(D_alt_top)
                    I_tops.append(I_top)
                    I_alt_tops.append(I_alt_top)
                D = np.array(D_tops)
                D_alt = np.array(D_alt_tops)
                I = np.array(I_tops)
                I_alt = np.array(I_alt_tops)

            # save the relevance scores for later analysis 
            for idx in range(len(batch_idxs)):
                ex_idx = batch_idxs[idx]
                hop1relevance_domain0[ex_idx] = {'scores': D[idx], 'docids': I[idx]}
                if D_alt is not None:
                    hop1relevance_domain1[ex_idx] = {'scores': D_alt[idx], 'docids': I_alt[idx]}

            # collect domain initial documents
            query_pairs = []
            for b_idx in range(bsize):
                for _, doc_id in enumerate(I[b_idx]):
                    if doc_id >= 0:
                        doc = id2doc[str(doc_id)]["text"]
                        if "roberta" in  args.model_name and doc.strip() == "":
                            doc = id2doc[str(doc_id)]["title"]
                            D[b_idx][_] = float("-inf")
                        query_pairs.append((batch_q[b_idx], doc))
                    else:
                        # we want to null out the document id so that we don't use it to retrieve in the next round
                        query_pairs.append(("-", "-"))

            if D_alt is not None:
                query_pairs_alt = []
                for b_idx in range(bsize):
                    for _, doc_id in enumerate(I_alt[b_idx]):
                        if doc_id >= 0:
                            if args.indexpath_alt:
                                doc_alt = id2doc_alt[str(doc_id)]["text"]
                            else:
                                doc_alt = id2doc[str(doc_id)]["text"]
                            if "roberta" in  args.model_name and doc_alt.strip() == "":
                                if args.indexpath_alt:
                                    doc_alt = id2doc_alt[str(doc_id)]["title"]
                                else:
                                    doc_alt = id2doc[str(doc_id)]["title"]
                                D_alt[b_idx][_] = float("-inf")
                            query_pairs_alt.append((batch_q[b_idx], doc_alt))
                        else:
                            # we want to null out the document id so that we don't use it to retrieve in the next round
                            query_pairs_alt.append(("-", "-"))

            # PRODUCE NEW ENCODINGS FOR NEXT HOP
            batch_q_sp_encodes = tokenizer.batch_encode_plus(query_pairs, max_length=args.max_q_sp_len, pad_to_max_length=True, return_tensors="pt")
            batch_q_sp_encodes = move_to_cuda(dict(batch_q_sp_encodes))
            q_sp_embeds = model.encode_q(batch_q_sp_encodes["input_ids"], batch_q_sp_encodes["attention_mask"], batch_q_sp_encodes.get("token_type_ids", None))
            q_sp_embeds = q_sp_embeds.contiguous().cpu().numpy()
            if args.hnsw:
                q_sp_embeds = convert_hnsw_query(q_sp_embeds)

            if D_alt is not None:
                batch_q_sp_encodes_alt = tokenizer.batch_encode_plus(query_pairs_alt, max_length=args.max_q_sp_len, pad_to_max_length=True, return_tensors="pt")
                batch_q_sp_encodes_alt = move_to_cuda(dict(batch_q_sp_encodes_alt))
                q_sp_embeds_alt = model.encode_q(batch_q_sp_encodes_alt["input_ids"], batch_q_sp_encodes_alt["attention_mask"], batch_q_sp_encodes_alt.get("token_type_ids", None))
                q_sp_embeds_alt = q_sp_embeds_alt.contiguous().cpu().numpy()
            
            # HOP 2
            # index over index 1 keys
            D_, I_ = index.search(q_sp_embeds, args.beam_size)
            D_ = D_.reshape(bsize, args.beam_size, args.beam_size)
            I_ = I_.reshape(bsize, args.beam_size, args.beam_size)
            
            if D_alt is not None:
                # index_alt over index 2 keys
                if args.indexpath_alt:
                    D_alt_, I_alt_ = index_alt.search(q_sp_embeds_alt, args.beam_size)
                    D_alt_ = D_alt_.reshape(bsize, args.beam_size, args.beam_size)
                    I_alt_ = I_alt_.reshape(bsize, args.beam_size, args.beam_size)
                elif args.retrieval_mode != "fullindex":
                    D_alt_, I_alt_ = index.search(q_sp_embeds_alt, args.beam_size)
                    D_alt_ = D_alt_.reshape(bsize, args.beam_size, args.beam_size)
                    I_alt_ = I_alt_.reshape(bsize, args.beam_size, args.beam_size)

                # index over index 2 keys
                D_swap_, I_swap_ = index.search(q_sp_embeds_alt, args.beam_size)
                D_swap_ = D_swap_.reshape(bsize, args.beam_size, args.beam_size)
                I_swap_ = I_swap_.reshape(bsize, args.beam_size, args.beam_size)

                # index_alt over index 1 keys
                if args.indexpath_alt:
                    D_alt_swap_, I_alt_swap_ = index_alt.search(q_sp_embeds, args.beam_size)
                    D_alt_swap_ = D_alt_swap_.reshape(bsize, args.beam_size, args.beam_size)
                    I_alt_swap_ = I_alt_swap_.reshape(bsize, args.beam_size, args.beam_size)
                elif args.retrieval_mode != "fullindex":
                    D_alt_swap_, I_alt_swap_ = index.search(q_sp_embeds, args.beam_size)
                    D_alt_swap_ = D_alt_swap_.reshape(bsize, args.beam_size, args.beam_size)
                    I_alt_swap_ = I_alt_swap_.reshape(bsize, args.beam_size, args.beam_size)

            # aggregate path scores
            path_scores_00 = np.expand_dims(D, axis=2) + D_              # index 1 with query --> results --> index 1 with results from index 1 --> results 
            if D_alt is not None:
                path_scores_01 = np.expand_dims(D, axis=2) + D_alt_swap_ # index 1 with query --> results --> index 2 with results from index 1 --> results 
                path_scores_10 = np.expand_dims(D_alt, axis=2) + D_swap_ # index 2 with query --> results --> index 1 with results from index 2 --> results 
                path_scores_11 = np.expand_dims(D_alt, axis=2) + D_alt_  # index 2 with query --> results --> index 2 with results from index 2 --> results 
                if args.hnsw:
                    path_scores = - path_scores

            # save the relevance scores for later analysis 
            for idx in range(len(batch_idxs)):
                ex_idx = batch_idxs[idx]
                hop2relevance_domain1_domain1[ex_idx] = {'scores': path_scores_00[idx], 'docids': I_[idx]}
                if D_alt is not None:
                    hop2relevance_domain2_domain2[ex_idx] = {'scores': path_scores_11[idx], 'docids': I_alt_[idx]}
                    hop2relevance_domain2_domain1[ex_idx] = {'scores': path_scores_10[idx], 'docids': I_swap_[idx]}
                    hop2relevance_domain1_domain2[ex_idx] = {'scores': path_scores_01[idx], 'docids': I_alt_swap_[idx]}

            # iterate through each query
            for idx in range(bsize):
                ds_idx = batch_idxs[idx]
                sp = batch_ann[idx]["sp"]
                search_scores_00 = path_scores_00[idx]
                if args.retrieval_mode != "fullindex":
                    search_scores_01 = path_scores_01[idx]
                    search_scores_10 = path_scores_10[idx]
                    search_scores_11 = path_scores_11[idx]
                    
                    # privacy restrictions
                    if args.doc_privacy:
                        # cannot allow private to public retrieval paths
                        search_scores_10 = np.full_like(search_scores_10, float("-inf"))
                    if args.query_privacy:
                        # the only paths allowed are private to private
                        search_scores_00 = np.full_like(search_scores_00, float("-inf"))
                        search_scores_10 = np.full_like(search_scores_10, float("-inf"))
                        search_scores_01 = np.full_like(search_scores_01, float("-inf"))
                    

                # np.argsort returns the indices that would sort the array
                # np.ravel() flattens the array
                # np.unravel_index uses these indices to find spots within a 100 x 100 matrix (or whatever provided dims)
                ranked_pairs_00 = np.vstack(np.unravel_index(np.argsort(search_scores_00.ravel())[::-1],(args.beam_size, args.beam_size))).transpose()
                if args.retrieval_mode == "4combo_separaterank" or args.retrieval_mode =="2combo_oracledomain":
                    ranked_pairs_01 = np.vstack(np.unravel_index(np.argsort(search_scores_01.ravel())[::-1],(args.beam_size, args.beam_size))).transpose()
                    ranked_pairs_10 = np.vstack(np.unravel_index(np.argsort(search_scores_10.ravel())[::-1],(args.beam_size, args.beam_size))).transpose()
                    ranked_pairs_11 = np.vstack(np.unravel_index(np.argsort(search_scores_11.ravel())[::-1],(args.beam_size, args.beam_size))).transpose()

                elif args.retrieval_mode == "4combo_overallrank":
                    combined_scores = search_scores_00.ravel()
                    combined_scores = np.append(combined_scores, search_scores_01.ravel())
                    combined_scores = np.append(combined_scores, search_scores_10.ravel())
                    combined_scores = np.append(combined_scores, search_scores_11.ravel())
                    ranked_pairs_00 = np.vstack(np.unravel_index(sort_scores_by_hop1_idx,(4*args.beam_size, args.beam_size))).transpose()

                elif args.retrieval_mode != "fullindex":
                    assert 0, "invalid retrieval mode"
                
                retrieved_titles = []
                hop1_titles = []
                paths, path_titles = [], []
                candidaite_chains = []
                if has_domains:
                    doms = batch_domains[idx]
                if args.retrieval_mode == "fullindex":
                    for kval in range(args.topk):
                        path_ids = ranked_pairs_00[kval]
                        first_I = I
                        firstcorpus = id2doc
                        second_I = I_
                        secondcorpus = id2doc

                        hop_1_id = first_I[idx, path_ids[0]]
                        hop_2_id = second_I[idx, path_ids[0], path_ids[1]]
                        retrieved_titles.append(firstcorpus[str(hop_1_id)]["title"])
                        retrieved_titles.append(secondcorpus[str(hop_2_id)]["title"])

                        path = [str(hop_1_id), str(hop_2_id)]
                        paths.append(path)
                        path_titles.append([firstcorpus[str(hop_1_id)]["title"], secondcorpus[str(hop_2_id)]["title"]])
                        hop1_titles.append(firstcorpus[str(hop_1_id)]["title"])

                        if title2sents_map:
                            ent_1 = get_candidate_sentences(firstcorpus[path[0]].copy(), title2sents_map)
                            ent_2 = get_candidate_sentences(secondcorpus[path[1]].copy(), title2sents_map)
                        candidaite_chains.append([ent_1, ent_2])

                elif args.retrieval_mode == "4combo_overallrank":
                    kval = 0
                    while(len(path_titles)) < args.topk and kval < len(ranked_pairs_00):
                        path_ids = ranked_pairs_00[kval]

                        first_I = I
                        firstcorpus = id2doc
                        second_I = I_
                        secondcorpus = id2doc
                        if path_ids[0] < args.beam_size:
                            # came from search scores 00
                            pass
                        elif path_ids[0] < args.beam_size*2:
                            # came from search scores 01
                            second_I = I_alt_swap_
                            if id2doc_alt:
                                secondcorpus = id2doc_alt
                        elif path_ids[0] < args.beam_size*3:
                            # came from search scores 10
                            first_I = I_alt
                            if id2doc_alt:
                                firstcorpus = id2doc_alt
                            second_I = I_swap_
                        else:
                            # came from search scores 11
                            first_I = I_alt
                            if id2doc_alt:
                                firstcorpus = id2doc_alt
                            second_I = I_alt_
                            if id2doc_alt:
                                secondcorpus = id2doc_alt

                        path_ids[0] = path_ids[0] % args.beam_size
                        hop_1_id = first_I[idx, path_ids[0]]
                        hop_2_id = second_I[idx, path_ids[0], path_ids[1]]
                        title1 = firstcorpus[str(hop_1_id)]["title"]
                        title2 = secondcorpus[str(hop_2_id)]["title"]
                        new_path = [str(hop_1_id), str(hop_2_id)]
                        new_path_titles = [title1, title2]

                        if new_path_titles not in path_titles:
                            retrieved_titles.append(title1)
                            retrieved_titles.append(title2)
                            paths.append(new_path)
                            path_titles.append(new_path_titles)
                            hop1_titles.append(title1)

                            if title2sents_map:
                                ent_1 = get_candidate_sentences(firstcorpus[new_path[0]].copy(), title2sents_map)
                                ent_2 = get_candidate_sentences(secondcorpus[new_path[1]].copy(), title2sents_map)
                            candidaite_chains.append([ent_1, ent_2])

                        kval += 1

                elif args.retrieval_mode == "4combo_separaterank":
                    idx_pairs = [(I, I_), (I, I_alt_swap_), (I_alt, I_swap_), (I_alt, I_alt_)]
                    corpus_pairs = [(id2doc, id2doc), (id2doc, id2doc_alt), (id2doc_alt, id2doc), (id2doc_alt, id2doc_alt)]
                    if id2doc_alt == None:
                        # this case tests with domain1 and domain2 being the same
                        corpus_pairs = [(id2doc, id2doc), (id2doc, id2doc), (id2doc, id2doc), (id2doc, id2doc)]
                    ranked_pairs = [ranked_pairs_00, ranked_pairs_01, ranked_pairs_10, ranked_pairs_11]
                    kval = 0
                    while len(candidaite_chains) < args.topk:
                        for ival in range(len(ranked_pairs)):
                            path_ids = ranked_pairs[ival][kval]
                            first_I = idx_pairs[ival][0]
                            
                            firstcorpus = corpus_pairs[ival][0]
                            second_I = idx_pairs[ival][1]
                            secondcorpus = corpus_pairs[ival][1]

                            hop_1_id = first_I[idx, path_ids[0]]
                            hop_2_id = second_I[idx, path_ids[0], path_ids[1]]

                            if hop_1_id < 0 or hop_2_id < 0:
                                continue

                            retrieved_titles.append(firstcorpus[str(hop_1_id)]["title"])
                            retrieved_titles.append(secondcorpus[str(hop_2_id)]["title"])

                            path = [str(hop_1_id), str(hop_2_id)]
                            paths.append(path)
                            path_titles.append([firstcorpus[str(hop_1_id)]["title"], secondcorpus[str(hop_2_id)]["title"]])
                            hop1_titles.append(firstcorpus[str(hop_1_id)]["title"])
                    
                            if title2sents_map:
                                ent_1 = get_candidate_sentences(firstcorpus[path[0]].copy(), title2sents_map)
                                ent_2 = get_candidate_sentences(secondcorpus[path[1]].copy(), title2sents_map)
                            candidaite_chains.append([ent_1, ent_2])
                        kval += 1

                        if ds_idx in indices:
                            import pdb;
                            pdb.set_trace()

                
                elif args.retrieval_mode =="2combo_oracledomain":
                    doms = batch_domains[idx]
                    for kval in range(args.topk):
                        first_I = I
                        firstcorpus = id2doc
                        second_I = I_
                        secondcorpus = id2doc
                        ranked_pairs = ranked_pairs_00
                        if doms[0] == 0 and doms[1] == 0:
                            pass
                        elif doms[0] == 0 and doms[1] == 1:
                            second_I = I_alt_swap_
                            if id2doc_alt:
                                secondcorpus = id2doc_alt
                            ranked_pairs = ranked_pairs_01
                        elif doms[0] == 1 and doms[1] == 0:
                            first_I = I_alt
                            if id2doc_alt:
                                firstcorpus = id2doc_alt
                            second_I = I_swap_
                            ranked_pairs = ranked_pairs_10
                        else:
                            first_I = I_alt
                            if id2doc_alt:
                                firstcorpus = id2doc_alt
                            second_I = I_alt_
                            if id2doc_alt:
                                secondcorpus = id2doc_alt
                            ranked_pairs = ranked_pairs_11

                        path_ids = ranked_pairs[kval]
                        hop_1_id = first_I[idx, path_ids[0]]
                        hop_2_id = second_I[idx, path_ids[0], path_ids[1]]
                        retrieved_titles.append(firstcorpus[str(hop_1_id)]["title"])
                        retrieved_titles.append(secondcorpus[str(hop_2_id)]["title"])

                        path = [str(hop_1_id), str(hop_2_id)]
                        paths.append(path)
                        path_titles.append([firstcorpus[str(hop_1_id)]["title"], secondcorpus[str(hop_2_id)]["title"]])
                        hop1_titles.append(firstcorpus[str(hop_1_id)]["title"])

                        if title2sents_map:
                            ent_1 = get_candidate_sentences(firstcorpus[path[0]].copy(), title2sents_map)
                            ent_2 = get_candidate_sentences(secondcorpus[path[1]].copy(), title2sents_map)
                        candidaite_chains.append([ent_1, ent_2])

                if args.only_eval_ans:
                    gold_answers = batch_ann[idx]["answer"]
                    concat_p = "yes no "
                    for i, p in enumerate(paths):
                        corpus = id2doc
                        if i == 0:
                            corpus = firstcorpus
                        elif i == 1:
                            corpus = secondcorpus
                        else:
                            assert 0, "path is longer than 2 hops... double check..."
                        concat_p += " ".join([corpus[doc_id]["title"] + " " + corpus[doc_id]["text"] for doc_id in p])
                    metrics.append({
                        "question": batch_ann[idx]["question"],
                        "ans_recall": int(para_has_answer(gold_answers, concat_p, simple_tokenizer)),
                        "type": batch_ann[idx].get("type", "single")
                    })
                else:
                    sp = batch_ann[idx]["sp"]
                    if type(sp[0]) != str:
                        new_sps = []
                        for s in sp:
                            if "_p" in s['title'] and "e" in s['title']:
                                new_sps.append(f"PERSONAL_{s['title']}")
                            else:
                                new_sps.append(s['title'])
                        sp = new_sps.copy()
                    assert len(set(sp)) == 2
                    type_ = batch_ann[idx]["type"]
                    question = batch_ann[idx]["question"]
                    p_recall, p_em = 0, 0
                    sp_covered = [sp_title in retrieved_titles for sp_title in sp]
                    if np.sum(sp_covered) > 0:
                        p_recall = 1
                    if np.sum(sp_covered) == len(sp_covered):
                        p_em = 1
                    path_covered = [int(set(p) == set(sp)) for p in path_titles]
                    path_covered = np.sum(path_covered) > 0
                    recall_1 = 0
                    covered_1 = [sp_title in hop1_titles for sp_title in sp]
                    if np.sum(covered_1) > 0: recall_1 = 1
                    metrics.append({
                        "_id": batch_ann[idx]["_id"],
                        "question": question,
                        "p_recall": p_recall,
                        "p_em": p_em,
                        "type": type_,
                        'recall_1': recall_1,
                        'path_covered': int(path_covered)
                    })

                    
                    retrieval_outputs.append({
                        "_id": batch_ann[idx]["_id"],
                        "question": question,
                        "candidate_chains": candidaite_chains,
                        # "sp": sp_chain,
                        # "answer": batch_ann[idx]["answer"],
                        # "type": type_,
                        # "coverd_k": covered_k
                    })

    if args.save_path != "" and not args.only_hop_1:
        with open(args.save_path, "w") as out:
            for l in retrieval_outputs:
                out.write(json.dumps(l) + "\n")
        with open(args.metrics_path, "w") as out:
            for m in metrics:
                out.write(json.dumps(m) + "\n")

    if not args.only_hop_1:
        logger.info(f"Evaluating {len(metrics)} samples...")
        type2items = collections.defaultdict(list)
        for item in metrics:
            type2items[item["type"]].append(item)
        if args.only_eval_ans:
            logger.info(f'Ans Recall: {np.mean([m["ans_recall"] for m in metrics])}')
            for t in type2items.keys():
                logger.info(f"{t} Questions num: {len(type2items[t])}")
                logger.info(f'Ans Recall: {np.mean([m["ans_recall"] for m in type2items[t]])}')
        else:
            logger.info(f'\tAvg PR: {np.mean([m["p_recall"] for m in metrics])}')
            logger.info(f'\tAvg P-EM: {np.mean([m["p_em"] for m in metrics])}')
            logger.info(f'\tAvg 1-Recall: {np.mean([m["recall_1"] for m in metrics])}')
            logger.info(f'\tPath Recall: {np.mean([m["path_covered"] for m in metrics])}')
            for t in type2items.keys():
                logger.info(f"{t} Questions num: {len(type2items[t])}")
                logger.info(f'\tAvg PR: {np.mean([m["p_recall"] for m in type2items[t]])}')
                logger.info(f'\tAvg P-EM: {np.mean([m["p_em"] for m in type2items[t]])}')
                logger.info(f'\tAvg 1-Recall: {np.mean([m["recall_1"] for m in type2items[t]])}')
                logger.info(f'\tPath Recall: {np.mean([m["path_covered"] for m in type2items[t]])}')

    # save the intermediate relevance scores
    logger.info("Saving the relevance scores for retrieval.")
    
    outfile = f"{os.path.dirname(args.save_path)}/{args.name}_hop1relevance_domain0.npz"
    np.savez(outfile, **hop1relevance_domain0)

    outfile = f"{os.path.dirname(args.save_path)}/{args.name}_hop1relevance_domain1.npz"
    np.savez(outfile, **hop1relevance_domain1)

    if not args.only_hop_1:
        outfile = f"{os.path.dirname(args.save_path)}/{args.name}_hop2relevance_domain1_domain1.npz"
        np.savez(outfile, **hop2relevance_domain1_domain1)

        outfile = f"{os.path.dirname(args.save_path)}/{args.name}_hop2relevance_domain1_domain2.npz"
        np.savez(outfile, **hop2relevance_domain1_domain2)

        outfile = f"{os.path.dirname(args.save_path)}/{args.name}_hop2relevance_domain2_domain2.npz"
        np.savez(outfile, **hop2relevance_domain2_domain2)

        outfile = f"{os.path.dirname(args.save_path)}/{args.name}_hop2relevance_domain2_domain1.npz"
        np.savez(outfile, **hop2relevance_domain2_domain1)

