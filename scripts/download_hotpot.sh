#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Make HotpotQA model folder. 
cd datasets
mkdir hotpotqa
cd hotpotqa

mkdir datasets
cd datasets
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot/hotpot_train_with_neg_v0.json
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot/hotpot_dev_with_neg_v0.json
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot/hotpot_qas_val.json
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot/train_retrieval_b100_k100_sp.json
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot/dev_retrieval_b50_k50_sp.json
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot/dev_retrieval_top100_sp.json

cd ..
mkdir corpora
cd corpora
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot_index/wiki_id2doc.json
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot_index/wiki_index.npy

cd ..
mkdir models
cd models
wget https://dl.fbaipublicfiles.com/mdpr/models/doc_encoder.pt
wget https://dl.fbaipublicfiles.com/mdpr/models/q_encoder.pt
wget https://dl.fbaipublicfiles.com/mdpr/models/qa_electra.pt

echo "Finished downloading models!"
