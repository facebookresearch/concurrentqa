TRAIN_DATA_PATH=/checkpoint/simarora/mdr/data/hotpot/
DEV_DATA_PATH=/checkpoint/simarora/mdr/data/hotpot/
RUN_ID=$1


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/train_mhop.py \
    --do_train \
    --prefix ${RUN_ID} \
    --predict_batch_size 3000 \
    --model_name roberta-base \
    --train_batch_size 150 \
    --learning_rate 2e-5 \
    --fp16 \
    --train_file ${TRAIN_DATA_PATH} \
    --predict_file ${DEV_DATA_PATH}  \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 300 \
    --max_q_len 70 \
    --max_q_sp_len 350 \
    --shared-encoder \
    --warmup-ratio 0.1
