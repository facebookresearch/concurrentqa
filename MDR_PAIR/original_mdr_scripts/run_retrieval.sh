# FULL RETRIEVAL
# python eval_mhop_retrieval.py \
#     /checkpoint/simarora/mdr/data/hotpot/hotpot_qas_val.json \
#     /checkpoint/simarora/mdr/data/hotpot_index/wiki_index.npy \
#     /checkpoint/simarora/mdr/data/hotpot_index/wiki_id2doc.json \
#     /checkpoint/simarora/mdr/models/q_encoder.pt \
#     --batch-size 100 \
#     --beam-size 1 \
#     --topk 1 \
#     --shared-encoder \
#     --model-name roberta-base \
#     --gpu \
#     --save-path /checkpoint/simarora/mdr/save_test_runs/save_eval_results.json


# DOMAIN - SPECIFIC RETRIEVAL
python eval_mhop_retrieval.py \
    /checkpoint/simarora/mdr/data/hotpot/global_qas_val.json \
    /checkpoint/simarora/mdr/data/hotpot_index/wiki_index.npy \
    /checkpoint/simarora/mdr/data/hotpot_index/wiki_id2doc.json \
    /checkpoint/simarora/mdr/models/q_encoder.pt \
    --batch-size 100 \
    --beam-size 1 \
    --topk 1 \
    --shared-encoder \
    --model-name roberta-base \
    --gpu \
    --save-path /checkpoint/simarora/mdr/save_test_runs/save_eval_results.json

    python eval_mhop_retrieval.py \
    /checkpoint/simarora/mdr/data/hotpot/hotpot_qas_val.json \
    /checkpoint/simarora/mdr/data/hotpot_index/wiki_index.npy \
    /checkpoint/simarora/mdr/data/hotpot_index/wiki_id2doc.json \
    /checkpoint/simarora/mdr/models/q_encoder.pt \
    --batch-size 100 \
    --beam-size 1 \
    --topk 1 \
    --shared-encoder \
    --model-name roberta-base \
    --gpu \
    --save-path /checkpoint/simarora/mdr/save_test_runs/save_eval_results.json
