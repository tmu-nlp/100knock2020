#!/bin/bash
set -eu
<< __EOF__
90. データの準備
機械翻訳のデータセットをダウンロードせよ．
訓練データ，開発データ，評価データを整形し，必要に応じてトークン化などの前処理を行うこと．
ただし，この段階ではトークンの単位として形態素（日本語）および単語（英語）を採用せよ．

[MEMO]
- https://cloud.google.com/tpu/docs/tutorials/transformer-pytorch?hl=ja

[Dataset]
- 京都フリー翻訳タスク (KFTT)
    - http://www.phontron.com/kftt/index-ja.html#dataonly

[Model - fairseq]
- Installation
    - https://github.com/pytorch/fairseq#requirements-and-installation
- Translation
    - https://github.com/pytorch/fairseq/tree/master/examples/translation#training-a-new-model
    - https://fairseq.readthedocs.io/en/latest/command_line_tools.html

[Usage]
bash knock90.sh
__EOF__

mkdir -p data
pushd data
    if ! [ -d kftt-data-1.0 ]; then
        wget http://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz
        tar -zxvf kftt-data-1.0.tar.gz
    fi
popd

EXP_NAME=KFTT.ja-en
rm -rf data-bin/$EXP_NAME

# Preprocess/binarize the data
# OOM 対策で threshold=5
TEXT=data/kftt-data-1.0/data/tok
fairseq-preprocess --source-lang ja --target-lang en \
    --trainpref $TEXT/kyoto-train \
    --validpref $TEXT/kyoto-dev \
    --testpref $TEXT/kyoto-test \
    --destdir data-bin/$EXP_NAME \
    --thresholdsrc 5 \
    --thresholdtgt 5 \
    --workers 20
