#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# FILL IN: ``subfolder'' should be uniquely named according to your experiment/run
SUBFOLDER=base

# FILL IN: modify ``base dir'' according to your directory path
BASE_DIR=/mnt/disks/scratch1/concurrentqa/
PREFIX_DIR=${BASE_DIR}/RUNS/
HOTPOT_MODELS_PATH=${BASE_DIR}/datasets/hotpotqa/models/

# train the retriever on CQA
TRAIN_DATA_PATH=${BASE_DIR}/datasets/concurrentqa/data/Retriever_CQA_train_all_original.json
DEV_DATA_PATH=${BASE_DIR}/datasets/concurrentqa/data/Retriever_CQA_dev_all_original.json

# retrieve and read
EVAL_TRAIN_DATA_PATH=${BASE_DIR}/datasets/concurrentqa/data/CQA_train_all.json
EVAL_DEV_DATA_PATH=${BASE_DIR}/datasets/concurrentqa/data/CQA_dev_all.json
EVAL_TEST_DATA_PATH=${BASE_DIR}/datasets/concurrentqa/data/CQA_test_all.json
TITLE2SENT_MAP=${BASE_DIR}/datasets/concurrentqa/corpora/title2sent_map.json

RUN_ID=0
NUM_RETRIEVED=100
DOC_PRIVACY=0
RETRIEVAL_MODE=fullindex
MIN_BY_DOMAIN=0 

# FILL IN: Set 1 (Yes) and 0 (No) for the steps you want to run.
TRAIN_CQA_DPR=1 # train a mdr model on the data at TRAIN_DATA_PATH
ENCODE_PASSAGE_CORPUS=1 # encode the corpora using the mdr model just trained
RETRIEVE_W_CQA_DPR=1 # retrieve using the trained mdr model from the encoded corpus
TRAIN_and_READ_CQA_ELECTRA=0 # train and read an electra reader model on cqa data
READ_DPR_RETRIEVED_W_ELECTRA_HOTPOT=1 # read using a pretrained electra model that was trained on hotpot data

################################################################
# RETRIEVAL to TRAIN EWQ-DPR
################################################################
MAX_C_LEN=300
MAX_Q_LEN=70
MAX_Q_SP_LEN=350

if [[ $TRAIN_CQA_DPR == 1 ]]; then
    CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 python train_mhop.py \
        --do_train \
        --prefix ${RUN_ID} \
        --predict_batch_size 3000 \
        --model_name roberta-base \
        --train_batch_size 150 \
        --learning_rate 5e-5 \
        --fp16 \
        --train_file ${TRAIN_DATA_PATH} \
        --predict_file ${DEV_DATA_PATH}  \
        --seed 16 \
        --eval-period -1 \
        --max_c_len ${MAX_C_LEN} \
        --max_q_len ${MAX_Q_LEN} \
        --max_q_sp_len ${MAX_Q_SP_LEN} \
        --shared-encoder \
        --warmup-ratio 0.1 \
        --output_dir ./logs/${SUBFOLDER}
fi

################################################################
#   INDEX THE WIKI AND ENRON PSGS TOGETHER
################################################################

# TODO: modify for your runs
if [[ "$SUBFOLDER" == run_1 ]]; then
    DPR_CQA_PATH=/03-02-2022/0-seed16-bsz150-fp16True-lr5e-05-decay0.0-warm0.1-valbsz3000-sharedTrue-multi1-schemenone/checkpoint_best.pt
    MODEL_CHECKPOINT=./logs/${SUBFOLDER}/${DPR_CQA_PATH}
fi

if [[ $ENCODE_PASSAGE_CORPUS == 1 ]]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 python encode_corpus.py \
        --do_predict \
        --predict_batch_size 1000 \
        --model_name roberta-base \
        --predict_file ${BASE_DIR}/datasets/concurrentqa/corpora/combined_corpus.json \
        --init_checkpoint ${MODEL_CHECKPOINT} \
        --embed_save_path ${PREFIX_DIR}/${SUBFOLDER} \
        --fp16 \
        --max_c_len ${MAX_C_LEN} \
        --num_workers 20 
fi

################################################################
#   PERFORM RETRIEVAL ON THE DEV SET USING EWQ-DPR
################################################################
if [[ $RETRIEVE_W_CQA_DPR == 1 ]]; then
    echo "Current experiment: $SUBFOLDER"
 
    # loop over train split too if we want to eventually train electra
    MODEL_NAME=cqa
    for split in dev test; do 
        if [[ "$split" == "train" ]]; then
            echo "'train' running retrieval with dpr model trained on cqa"
            DPR_CQA_DATA=${EVAL_TRAIN_DATA_PATH}
        elif [[ "$split" == "test" ]]; then
            echo "'test' running retrieval with dpr model trained on cqa"
            DPR_CQA_DATA=${EVAL_TEST_DATA_PATH}
        elif [[ "$split" == "dev" ]]; then 
            echo "'dev' running retrieval with dpr model trained on cqa"
            DPR_CQA_DATA=${EVAL_DEV_DATA_PATH}
        fi

        #MODEL_CHECKPOINT=//mnt/disks/scratch1/hotpot_models/q_encoder.pt
        CUDA_VISIBLE_DEVICES=4,5,6,7 python eval_mhop_retrieval.py \
            ${DPR_CQA_DATA} \
            ${PREFIX_DIR}/${SUBFOLDER}/idx.npy \
            ${PREFIX_DIR}/${SUBFOLDER}/id2doc.json \
            ${MODEL_CHECKPOINT} \
            --batch-size 12 \
            --beam-size ${NUM_RETRIEVED} \
            --topk ${NUM_RETRIEVED} \
            --shared-encoder \
            --model-name roberta-base \
            --gpu \
            --gpu_num 2 \
            --name ${split}_num${NUM_RETRIEVED}_mode${RETRIEVAL_MODE}_DOC_PRIVACY${DOC_PRIVACY}_minRETbyDomain${MIN_BY_DOMAIN}_model${MODEL_NAME} \
            --retrieval_mode ${RETRIEVAL_MODE} \
            --save-path ${PREFIX_DIR}${SUBFOLDER}/${split}_retrieval_results_num${NUM_RETRIEVED}_mode${RETRIEVAL_MODE}_DOC_PRIVACY${DOC_PRIVACY}_minRETbyDomain${MIN_BY_DOMAIN}_model${MODEL_NAME}.json \
            --metrics-path ${PREFIX_DIR}${SUBFOLDER}/${split}_retrieval_metrics_num${NUM_RETRIEVED}_mode${RETRIEVAL_MODE}_DOC_PRIVACY${DOC_PRIVACY}_minRETbyDomain${MIN_BY_DOMAIN}_model${MODEL_NAME}.json
    done
fi


################################################################
# ADD SP LABELS (IGNORE HOTPOT, TRAIN)
################################################################
if [[ $TRAIN_and_READ_CQA_ELECTRA == 1 ]] || [[ $READ_DPR_RETRIEVED_W_ELECTRA_HOTPOT == 1 ]]; then 
    echo "Adding SP Labels!!!"
    for MODEL_NAME in cqa;
    do
        for split in dev test; do 
            if [[ "$split" == "train" ]]; then
                echo "x has the value 'train'"
                ORIGINAL_DATA=${EVAL_TRAIN_DATA_PATH}
            elif [[ "$split" == "test" ]]; then
                echo "x has the value 'dev'"
                ORIGINAL_DATA=${EVAL_TEST_DATA_PATH}
            elif [[ "$split" == "dev" ]]; then
                echo "x has the value 'dev'"
                ORIGINAL_DATA=${EVAL_DEV_DATA_PATH}
            fi

            RETRIEVED_DATA=${PREFIX_DIR}${SUBFOLDER}/${split}_retrieval_results_num${NUM_RETRIEVED}_mode${RETRIEVAL_MODE}_DOC_PRIVACY${DOC_PRIVACY}_minRETbyDomain${MIN_BY_DOMAIN}_model${MODEL_NAME}.json
            SAVED_PATH=${PREFIX_DIR}${SUBFOLDER}/${split}_retrieval_results_num${NUM_RETRIEVED}_mode${RETRIEVAL_MODE}_DOC_PRIVACY${DOC_PRIVACY}_minRETbyDomain${MIN_BY_DOMAIN}_model${MODEL_NAME}_w_sp.json
            python mdr/retrieval/utils/mhop_utils.py ${ORIGINAL_DATA} ${RETRIEVED_DATA} ${SAVED_PATH} ${TITLE2SENT_MAP}
        done
    done
fi

# ###########################################################################
# TRAINING ELECTRA-CQA MODEL USING PSGS RETRIEVED WITH DPR-CQA
# ###########################################################################
if [[ $TRAIN_and_READ_CQA_ELECTRA == 1 ]]; then
    # RUN QA READING on EACH INDEX (SINGLE-RETRIEVAL)
    for MODEL_NAME in cqa;
    do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_qa.py \
            --do_train \
            --prefix electra_large_debug_sn \
            --predict_batch_size 1024 \
            --model_name google/electra-large-discriminator \
            --train_batch_size 12 \
            --learning_rate 5e-5 \
            --train_file ${PREFIX_DIR}${SUBFOLDER}/train_retrieval_results_num${NUM_RETRIEVED}_mode${RETRIEVAL_MODE}_DOC_PRIVACY${DOC_PRIVACY}_minRETbyDomain${MIN_BY_DOMAIN}_model${MODEL_NAME}_w_sp.json \
            --predict_file ${PREFIX_DIR}${SUBFOLDER}/dev_retrieval_results_num${NUM_RETRIEVED}_mode${RETRIEVAL_MODE}_DOC_PRIVACY${DOC_PRIVACY}_minRETbyDomain${MIN_BY_DOMAIN}_model${MODEL_NAME}_w_sp.json \
            --output_dir ./logs/${SUBFOLDER}/cqa/ \
            --seed 42 \
            --eval-period 250 \
            --max_seq_len 512 \
            --max_q_len 64 \
            --gradient_accumulation_steps 8 \
            --neg-num 5 \
            --fp16 \
            --use-adam \
            --warmup-ratio 0.1 \
            --sp-weight 0.05 \
            --sp-pred 

    # TODO: modify this according to your runs
    echo "Current experiment: $SUBFOLDER"
    if [[ "$SUBFOLDER" == run_1 ]]; then
        ELTRA_CQA_PATH=/cqa/03-02-2022/electra_large_debug_sn-seed42-bsz12-fp16True-lr5e-05-decay0.0-neg5-snFalse-adamTrue-warm0.1-sp0.05-rank-1/checkpoint_best.pt 
        CQA_ELECTRA_MODEL_CKPT=${BASE_DIR}/MDR/logs/${SUBFOLDER}/${ELTRA_CQA_PATH}
    fi
    
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_qa.py \
        --do_predict \
        --predict_batch_size 200 \
        --model_name google/electra-large-discriminator \
        --fp16 \
        --predict_file ${PREFIX_DIR}${SUBFOLDER}/test_retrieval_results_num${NUM_RETRIEVED}_mode${RETRIEVAL_MODE}_DOC_PRIVACY${DOC_PRIVACY}_minRETbyDomain${MIN_BY_DOMAIN}_model${MODEL_NAME}_w_sp.json \
        --max_seq_len 512 \
        --max_q_len 64 \
        --init_checkpoint ${CQA_ELECTRA_MODEL_CKPT} \
        --sp-pred \
        --max_ans_len 30 \
        --save-prediction ${PREFIX_DIR}${SUBFOLDER}/test_enron_wiki_result_num${NUM_RETRIEVED}_mode${RETRIEVAL_MODE}_DOC_PRIVACY${DOC_PRIVACY}_minRETbyDomain${MIN_BY_DOMAIN}.json \
        --save_raw_results ${PREFIX_DIR}${SUBFOLDER}/test_enronwiki_reader_rawanswers_num${NUM_RETRIEVED}_mode${RETRIEVAL_MODE}_DOC_PRIVACY${DOC_PRIVACY}_minRETbyDomain${MIN_BY_DOMAIN}.json
    done
fi

# #################################################################################################
# READING WITH ELECTRA MODEL TRAINED ON HOTPOT, USING PSGS RETRIEVED WITH RETRIEVER TRAINED ON CQA
# #################################################################################################
if [[ $READ_DPR_RETRIEVED_W_ELECTRA_HOTPOT == 1 ]]; then
    for MODEL_NAME in cqa;
    do
        for split in dev test;
        do
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_qa.py \
                --do_predict \
                --predict_batch_size 200 \
                --model_name google/electra-large-discriminator \
                --fp16 \
                --predict_file ${PREFIX_DIR}${SUBFOLDER}/${split}_retrieval_results_num${NUM_RETRIEVED}_mode${RETRIEVAL_MODE}_DOC_PRIVACY${DOC_PRIVACY}_minRETbyDomain${MIN_BY_DOMAIN}_model${MODEL_NAME}_w_sp.json \
                --max_seq_len 512 \
                --max_q_len 64 \
                --init_checkpoint ${HOTPOT_MODELS_PATH}/qa_electra.pt \
                --sp-pred \
                --max_ans_len 30 \
                --output_dir ./logs/${SUBFOLDER}/hotpot/ \
                --save-prediction ${PREFIX_DIR}${SUBFOLDER}/${split}_enron_wiki_result_num${NUM_RETRIEVED}_mode${RETRIEVAL_MODE}_DOC_PRIVACY${DOC_PRIVACY}_minRETbyDomain${MIN_BY_DOMAIN}_model${MODEL_NAME}.json \
                --save_raw_results ${PREFIX_DIR}${SUBFOLDER}/${split}_enronwiki_reader_rawanswers_num${NUM_RETRIEVED}_mode${RETRIEVAL_MODE}_DOC_PRIVACY${DOC_PRIVACY}_minRETbyDomain${MIN_BY_DOMAIN}_model${MODEL_NAME}.json
        done
    done
fi
