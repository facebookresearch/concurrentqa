#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# FILL IN: ``subfolder'' should be uniquely named according to your experiment/run
SUBFOLDER=doc_privacy

# FILL IN: modify ``base dir'' according to your directory path
BASE_DIR=/mnt/disks/scratch1/concurrentqa/
PREFIX_DIR=${BASE_DIR}/RUNS/
HOTPOT_MODELS_PATH=${BASE_DIR}/datasets/hotpotqa/models/

TRAIN_DATA_PATH=${BASE_DIR}/datasets/concurrentqa/data/Retriever_CQA_train_all_original.json
DEV_DATA_PATH=${BASE_DIR}/datasets/concurrentqa/data/Retriever_CQA_dev_all_original.json
EVAL_TRAIN_DATA_PATH=${BASE_DIR}/datasets/concurrentqa/data/CQA_train_all.json
EVAL_DEV_DATA_PATH=${BASE_DIR}/datasets/concurrentqa/data/CQA_dev_all.json
TITLE2SENT_MAP=${BASE_DIR}/datasets/concurrentqa/corpora/title2sent_map.json

# FILL IN: Settings
RUN_ID=0
NUM_RETRIEVED=100 # k, retrieve top k documents per hop
DOC_PRIVACY=1 # document privacy; restrict private hop 1 to public hop 2 retrieval path
QUERY_PRIVACY=0 # query privacy; only retrieve from local corpus
MIN_BY_DOMAIN=0 # assert a requirement to retrieve at least this number of passages per (public and private) corpus

# FILL IN: Set 1 (Yes) and 0 (No) for the steps you want to run.
ENCODE_PASSAGE_CORPUS=1 # index all the public and private documents
RETRIEVE_W_HOTPOT_DPR=1 # retrieve from public and private indices
READ_HOTPOT_RETRIEVED_W_ELECTRA_HOTPOT=1 # read from retrieved documents

# FILL IN: RETRIEVAL MODES
# 4combo_separaterank: private and public documents are in 2 different corpora, retrieve separately and take the top k passage-chains from each of public/private, private/public, private/private, public/public chains for the reader
# 4combo_overallrank: private and public documents are in 2 different corpora, retrieve separately and take the top k passage-chains overall public/private, private/public, private/private, public/public chains for the reader
# fullindex: treat all private and public documents in one corpus and retrieve
# 2combo_oracledomain: retrieve with the knowledge of the gold supporting paths
RETRIEVAL_MODE=4combo_overallrank

################################################################
# ENRON INDEXING CORPUS
################################################################
if [[ $ENCODE_PASSAGE_CORPUS == 1 ]]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3 python encode_corpus.py \
     --do_predict \
     --predict_batch_size 1000 \
     --model_name roberta-base \
     --predict_file ${BASE_DIR}/datasets/concurrentqa/corpora/combined_corpus.json \
     --init_checkpoint  ${HOTPOT_MODELS_PATH}/doc_encoder.pt \
     --embed_save_path ${PREFIX_DIR}/${SUBFOLDER}/combinedcorpus \
     --fp16 \
     --max_c_len 300 \
     --num_workers 20 
     
  CUDA_VISIBLE_DEVICES=0,1,2,3 python encode_corpus.py \
     --do_predict \
     --predict_batch_size 1000 \
     --model_name roberta-base \
     --predict_file ${BASE_DIR}/datasets/concurrentqa/corpora/enron_corpus.json \
     --init_checkpoint ${HOTPOT_MODELS_PATH}/doc_encoder.pt \
     --embed_save_path ${PREFIX_DIR}/${SUBFOLDER}/enrononly \
     --fp16 \
     --max_c_len 300 \
     --num_workers 20 
     
  CUDA_VISIBLE_DEVICES=0,1,2,3 python encode_corpus.py \
     --do_predict \
     --predict_batch_size 1000 \
     --model_name roberta-base \
     --predict_file ${BASE_DIR}/datasets/concurrentqa/corpora/wiki_corpus.json \
     --init_checkpoint  ${HOTPOT_MODELS_PATH}/doc_encoder.pt \
     --embed_save_path ${PREFIX_DIR}/${SUBFOLDER}/wikionly \
     --fp16 \
     --max_c_len 300 \
     --num_workers 20 
fi

################################################################
#   PERFORM RETRIEVAL ON THE DEV SET USING HOTPOT-DPR
################################################################
if [[ $RETRIEVE_W_HOTPOT_DPR == 1 ]] && [[ $RETRIEVAL_MODE != "fullindex" ]]; then
  for split in dev; do
      echo "Current experiment: $SUBFOLDER"
      CUDA_VISIBLE_DEVICES=0,1 python eval_mhop_retrieval.py \
        ${EVAL_DEV_DATA_PATH} \
        ${PREFIX_DIR}/${SUBFOLDER}/wikionly/idx.npy \
        ${PREFIX_DIR}/${SUBFOLDER}/wikionly/id2doc.json \
        ${HOTPOT_MODELS_PATH}/q_encoder.pt \
        --indexpath_alt ${PREFIX_DIR}/${SUBFOLDER}/enrononly/idx.npy \
        --corpus_dict_alt ${PREFIX_DIR}/${SUBFOLDER}/enrononly/id2doc.json \
        --batch-size 12 \
        --beam-size ${NUM_RETRIEVED} \
        --topk ${NUM_RETRIEVED} \
        --shared-encoder \
        --model-name roberta-base \
        --retrieval_mode ${RETRIEVAL_MODE} \
        --doc_privacy ${DOC_PRIVACY} \
        --query_privacy ${QUERY_PRIVACY} \
        --min_retrieved_by_domain ${MIN_BY_DOMAIN} \
        --gpu \
        --gpu_num 1 \
        --name ${split}_num${NUM_RETRIEVED}_qpriv${QUERY_PRIVACY}_mode${RETRIEVAL_MODE}_DOC_PRIVACY${DOC_PRIVACY}_minRETbyDomain${MIN_BY_DOMAIN} \
        --save-path ${PREFIX_DIR}${SUBFOLDER}/${split}_retrieval_results_num${NUM_RETRIEVED}_qpriv${QUERY_PRIVACY}_mode${RETRIEVAL_MODE}_DOC_PRIVACY${DOC_PRIVACY}_minRETbyDomain${MIN_BY_DOMAIN}.json \
        --metrics-path ${PREFIX_DIR}${SUBFOLDER}/${split}_retrieval_metrics_num${NUM_RETRIEVED}_qpriv${QUERY_PRIVACY}_mode${RETRIEVAL_MODE}_DOC_PRIVACY${DOC_PRIVACY}_minRETbyDomain${MIN_BY_DOMAIN}.json
  done
fi

# retrieve from a combined corpus of the public and private documents, ignoring privacy concerns.
if [[ $RETRIEVE_W_HOTPOT_DPR == 1 ]] && [[ $RETRIEVAL_MODE == "fullindex" ]]; then
  for split in dev; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_mhop_retrieval.py \
        ${EVAL_DEV_DATA_PATH} \
        ${PREFIX_DIR}/${SUBFOLDER}/combinedcorpus/idx.npy \
        ${PREFIX_DIR}/${SUBFOLDER}/combinedcorpus/id2doc.json \
        ${HOTPOT_MODELS_PATH}/q_encoder.pt \
        --batch-size 12 \
        --beam-size ${NUM_RETRIEVED} \
        --topk ${NUM_RETRIEVED} \
        --shared-encoder \
        --model-name roberta-base \
        --retrieval_mode ${RETRIEVAL_MODE} \
        --gpu \
        --gpu_num 1 \
        --name ${split}_num${NUM_RETRIEVED}_qpriv${QUERY_PRIVACY}_mode${RETRIEVAL_MODE}_DOC_PRIVACY${DOC_PRIVACY}_minRETbyDomain${MIN_BY_DOMAIN} \
        --save-path ${PREFIX_DIR}${SUBFOLDER}/${split}_retrieval_results_num${NUM_RETRIEVED}_qpriv${QUERY_PRIVACY}_mode${RETRIEVAL_MODE}_DOC_PRIVACY${DOC_PRIVACY}_minRETbyDomain${MIN_BY_DOMAIN}.json \
        --metrics-path ${PREFIX_DIR}${SUBFOLDER}/${split}_retrieval_metrics_num${NUM_RETRIEVED}_qpriv${QUERY_PRIVACY}_mode${RETRIEVAL_MODE}_DOC_PRIVACY${DOC_PRIVACY}_minRETbyDomain${MIN_BY_DOMAIN}.json
  done
fi

###############################################################
# ADD SP LABELS AND READ WITH ELECTRA HOTPOT
###############################################################
if [[ $READ_HOTPOT_RETRIEVED_W_ELECTRA_HOTPOT == 1 ]]; then 
    # PERFORM QUERY ANSWERING
    for split in dev; do
        ORIGINAL_DATA=${EVAL_DEV_DATA_PATH}
        RETRIEVED_DATA=${PREFIX_DIR}${SUBFOLDER}/${split}_retrieval_results_num${NUM_RETRIEVED}_qpriv${QUERY_PRIVACY}_mode${RETRIEVAL_MODE}_DOC_PRIVACY${DOC_PRIVACY}_minRETbyDomain${MIN_BY_DOMAIN}.json
        SAVED_PATH=${PREFIX_DIR}${SUBFOLDER}/${split}_retrieval_results_num${NUM_RETRIEVED}_qpriv${QUERY_PRIVACY}_mode${RETRIEVAL_MODE}_DOC_PRIVACY${DOC_PRIVACY}_minRETbyDomain${MIN_BY_DOMAIN}_w_sp.json
        python mdr/retrieval/utils/mhop_utils.py ${ORIGINAL_DATA} ${RETRIEVED_DATA} ${SAVED_PATH} ${TITLE2SENT_MAP}
    done

    # RUN QA READING on EACH INDEX (SINGLE-RETRIEVAL)
    for split in dev; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_qa.py \
            --do_predict \
            --predict_batch_size 200 \
            --model_name google/electra-large-discriminator \
            --fp16 \
            --predict_file ${PREFIX_DIR}${SUBFOLDER}/${split}_retrieval_results_num${NUM_RETRIEVED}_qpriv${QUERY_PRIVACY}_mode${RETRIEVAL_MODE}_DOC_PRIVACY${DOC_PRIVACY}_minRETbyDomain${MIN_BY_DOMAIN}_w_sp.json \
            --max_seq_len 512 \
            --max_q_len 64 \
            --init_checkpoint ${HOTPOT_MODELS_PATH}/qa_electra.pt \
            --sp-pred \
            --max_ans_len 30 \
            --save-prediction ${PREFIX_DIR}${SUBFOLDER}/${split}_enron_wiki_result_num${NUM_RETRIEVED}_qpriv${QUERY_PRIVACY}_mode${RETRIEVAL_MODE}_DOC_PRIVACY${DOC_PRIVACY}_minRETbyDomain${MIN_BY_DOMAIN}.json \
            --save_raw_results ${PREFIX_DIR}${SUBFOLDER}/${split}_enronwiki_reader_rawanswers_num${NUM_RETRIEVED}_qpriv${QUERY_PRIVACY}_mode${RETRIEVAL_MODE}_DOC_PRIVACY${DOC_PRIVACY}_minRETbyDomain${MIN_BY_DOMAIN}.json
    done
fi

