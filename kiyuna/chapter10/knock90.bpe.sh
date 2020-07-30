#!/bin/bash
set -eu
<<__EOF__
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
bash knock90.bpe.sh
__EOF__

mkdir -p data
pushd data
if ! [ -d kftt-data-1.0 ]; then
    wget http://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz
    tar -zxvf kftt-data-1.0.tar.gz
fi
popd

SRC=data/kftt-data-1.0/data/tok
DST=data/kftt-data-1.0/data/bpe
BPEROOT=fairseq/examples/translation/subword-nmt/subword_nmt
BPE_TOKENS=10000
BPE_CODE=$DST/code

if ! [ -f $BPE_CODE ]; then
    echo "learn_bpe.py"
    mkdir -p $DST
    cat $SRC/kyoto-train.ja $SRC/kyoto-train.en >$SRC/kyoto-train.ja-en
    python $BPEROOT/learn_bpe.py -s $BPE_TOKENS <$SRC/kyoto-train.ja-en >$BPE_CODE
    echo "apply_bpe.py"
    for file in $(find $SRC -type f); do
        echo $(basename $file)
        python $BPEROOT/apply_bpe.py -c $BPE_CODE <$file >$DST/$(basename $file)
    done
fi

EXP_NAME=KFTT.bpe.ja-en
rm -rf data-bin/$EXP_NAME

TEXT=data/kftt-data-1.0/data/tok
fairseq-preprocess --source-lang ja --target-lang en \
    --trainpref $DST/kyoto-train \
    --validpref $DST/kyoto-dev \
    --testpref $DST/kyoto-test \
    --destdir data-bin/$EXP_NAME \
    --bpe subword_nmt \
    --workers 20
