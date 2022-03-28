# #!/bin/bash
# #SBATCH --cpus-per-task=5
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --gres=gpu:8
# #SBATCH --time=36:00:00
# #SBATCH --job-name=hotpot
# #SBATCH --output=/private/home/%u/pqa/FiD/%A
# #SBATCH --partition=devlab
# #SBATCH --mem=470GB
# #SBATCH --signal=USR1@140
# #SBATCH --open-mode=append
# #SBATCH --constraint=volta32gb
# #SBATCH --exclude=learnfair[5025-5300]

# export NGPU=8;
# port=$(shuf -i 15000-16000 -n 1)

# size=base
# ncontext=50
# name=$SLURM_JOB_ID"_"$size"_"$ncontext
# mp=none

# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch
# # ORIGINAL_DATA=/checkpoint/simarora/mdr/data/hotpot/hotpot_train_v1.1.json
# # RETRIEVED_DATA=/checkpoint/simarora/FiD/HotpotQA_FiD/ELECTRA_train_k100_1hop.json
# # SAVED_PATH=/checkpoint/simarora/FiD/HotpotQA_FiD/ELECTRA_train_k100_1hop_w_sp.json
# # python mdr/retrieval/utils/mhop_utils.py ${ORIGINAL_DATA} ${RETRIEVED_DATA} ${SAVED_PATH}


# # ORIGINAL_DATA=/checkpoint/simarora/mdr/data/hotpot/hotpot_dev_distractor_v1.json
# # RETRIEVED_DATA=/checkpoint/simarora/FiD/HotpotQA_FiD/ELECTRA_dev_k100_1hop.json
# # SAVED_PATH=/checkpoint/simarora/FiD/HotpotQA_FiD/ELECTRA_dev_k100_1hop_w_sp.json
# # python mdr/retrieval/utils/mhop_utils.py ${ORIGINAL_DATA} ${RETRIEVED_DATA} ${SAVED_PATH}


port=$(shuf -i 15000-16000 -n 1)
# srun ~/anaconda3/envs/venv071821/bin/python3 

for i in {0..7};
    do
        RANK=$i python -m torch.distributed.launch train_qa.py \
            --do_train \
            --prefix electra_large_debug_sn \
            --predict_batch_size 1024 \
            --model_name google/electra-large-discriminator \
            --train_batch_size 12 \
            --learning_rate 5e-5 \
            --train_file /checkpoint/simarora/FiD/HotpotQA_FiD/ELECTRA_dev_k100_1hop_w_sp.json \
            --predict_file /checkpoint/simarora/FiD/HotpotQA_FiD/ELECTRA_dev_k100_1hop_w_sp.json \
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
            --sp-pred \
            --local_rank $i&
done
    