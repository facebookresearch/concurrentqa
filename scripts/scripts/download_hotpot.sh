#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Make HotpotQA model folder. 
cd datasets

mkdir hotpotqa
cd hotpotqa

mkdir models
cd models

wget https://dl.fbaipublicfiles.com/mdpr/models/doc_encoder.pt
wget https://dl.fbaipublicfiles.com/mdpr/models/q_encoder.pt
wget https://dl.fbaipublicfiles.com/mdpr/models/qa_electra.pt

echo "Finished downloading models!"