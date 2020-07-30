#!/bin/bash
set -eu
<<__EOF__
92. 機械翻訳モデルの適用
91で学習したニューラル機械翻訳モデルを用い，
与えられた（任意の）日本語の文を英語に翻訳するプログラムを実装せよ．

[Usage]
sh knock92.bpe.sh 3 KFTT.bpe.ja-en
__EOF__

GPU=$1
EXP_NAME=$2

mkdir -p dumped

# 任意の日本語文に対応してなさそう
# CUDA_VISIBLE_DEVICES=$GPU fairseq-generate \
#     --path checkpoints/$EXP_NAME/checkpoint_best.pt \
#     data-bin/$EXP_NAME > dumped/knock92.$EXP_NAME.txt

SRC_PATH=data/kftt-data-1.0/data/tok/kyoto-test.ja
BPE_CODE=data/kftt-data-1.0/data/bpe/code

CUDA_VISIBLE_DEVICES=$GPU fairseq-interactive \
    --path checkpoints/$EXP_NAME/checkpoint_best.pt \
    --remove-bpe --bpe=subword_nmt --bpe-codes $BPE_CODE \
    data-bin/$EXP_NAME <$SRC_PATH | grep '^H' | cut -f3 >dumped/k92.$EXP_NAME.txt
