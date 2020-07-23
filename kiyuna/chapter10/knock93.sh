#!/bin/bash
set -eu
<< __EOF__
93. BLEUスコアの計測
91で学習したニューラル機械翻訳モデルの品質を調べるため，評価データにおけるBLEUスコアを測定せよ．

[Usage]
sh knock93.sh KFTT.ja-en
sh knock93.sh KFTT.mini.ja-en
sh knock93.sh KFTT.bpe.ja-en
__EOF__

EXP_NAME=$1

REF_PATH=data/kftt-data-1.0/data/tok/kyoto-test.en

fairseq-score \
    --sys dumped/k92.$EXP_NAME.txt \
    --ref $REF_PATH | tee dumped/k93.$EXP_NAME.score.txt
