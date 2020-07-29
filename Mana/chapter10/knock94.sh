#!/bin/bash -eu

# 94. ビーム探索
# 91で学習したニューラル機械翻訳モデルで翻訳文をデコードする際に，ビーム探索を導入せよ．ビーム幅を1から100くらいまで適当に変化させながら，開発セット上のBLEUスコアの変化をプロットせよ．

# 93とほとんど同じ。
# --beamの値を変化させる。可視化はtensorboardでやればいいのでは。

for N in `seq 1 100` ; do
    CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/kftt.ja-en  \
        --path knock91/checkpoint_best.pt \
        --beam $N --batch-size 128 --remove-bpe | tee /tmp/gen.out

    grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
    grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref

    fairseq-score --sys /tmp/gen.out.sys --ref /tmp/gen.out.ref >> knock94.score
done