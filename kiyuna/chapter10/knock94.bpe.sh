#!/bin/bash
set -eu
<<__EOF__
94. ビーム探索
91で学習したニューラル機械翻訳モデルで翻訳文をデコードする際に，ビーム探索を導入せよ．
ビーム幅を1から100くらいまで適当に変化させながら，開発セット上のBLEUスコアの変化をプロットせよ．

[Usage]
sh knock94.bpe.sh 4 KFTT.bpe.ja-en
__EOF__

GPU=$1
EXP_NAME=$2

mkdir -p dumped

SRC_PATH=data/kftt-data-1.0/data/tok/kyoto-test.ja
REF_PATH=data/kftt-data-1.0/data/tok/kyoto-test.en
BPE_CODE=data/kftt-data-1.0/data/bpe/code

rm -rf dumped/k94.$EXP_NAME.score.txt
for N in $(seq 1 10); do
    # knock92
    CUDA_VISIBLE_DEVICES=$GPU fairseq-interactive \
        --path checkpoints/$EXP_NAME/checkpoint_best.pt \
        --beam $N \
        --remove-bpe --bpe=subword_nmt --bpe-codes $BPE_CODE \
        data-bin/$EXP_NAME <$SRC_PATH | grep '^H' | cut -f3 >dumped/k94.$EXP_NAME.$N.txt
    # knock93
    fairseq-score \
        --sys dumped/k94.$EXP_NAME.$N.txt \
        --ref $REF_PATH |
        tail -n1 | cut -f3 -d" " | sed 's/,//g' >>dumped/k94.$EXP_NAME.score.txt
done

python plot.py dumped/k94.$EXP_NAME.score.txt
