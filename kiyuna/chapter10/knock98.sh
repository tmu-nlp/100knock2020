#!/bin/bash
set -eu
<<__EOF__
98. ドメイン適応
ニューラルネットワークのモデルや，そのハイパーパラメータを変更しつつ，
開発データにおけるBLEUスコアが最大となるモデルとハイパーパラメータを求めよ．

[Pre-trained model]
- https://github.com/MorinoseiMorizo/jparacrawl-finetune

# Docker
docker pull morinoseimorizo/jparacrawl-fairseq
HOST_DIR=/clwork/tomoshige/_playground/kiyuna/chapter10
docker run -it --gpus 5 -v $HOST_DIR:/host_disk morinoseimorizo/jparacrawl-fairseq bash

# Prepare the data
cd /host_disk
git clone https://github.com/MorinoseiMorizo/jparacrawl-finetune.git   # Clone the repository.
cd jparacrawl-finetune
./get-data.sh   # This script will download KFTT and sentencepiece model for pre-processing the corpus.
./preprocess.sh   # Split the corpus into subwords.
cp ./ja-en/*.sh ./   # If you try the English-to-Japanese example, use en-ja directory instead.
./get-model.sh   # Download the pre-trained model.

# Fine-tuning on KFTT corpus
cd /host_disk/jparacrawl-finetune
nohup ./fine-tune_kftt_fp32.sh 1 &> fine-tune.log

# Evaluation
cat models/fine-tune/test.log
__EOF__
