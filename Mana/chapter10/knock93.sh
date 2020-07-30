#!/bin/bash -eu

# 93. BLEUスコアの計測
# 91で学習したニューラル機械翻訳モデルの品質を調べるため，評価データにおけるBLEUスコアを測定せよ．
# 参考 https://github.com/pytorch/fairseq/tree/master/examples/translation

CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/kftt.ja-en  \
    --path knock91/checkpoint_best.pt \
    --beam 5 --batch-size 128 --remove-bpe | tee /tmp/gen.out

grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref

fairseq-score --sys /tmp/gen.out.sys --ref /tmp/gen.out.ref

# ^はbashの正規表現。
# こんな感じで出力される。
# BLEU4 = 0.84, 19.0/2.1/0.2/0.1 (BP=1.000, ratio=1.079, syslen=28850, reflen=26734)

# generateコマンド自体の出力はこんな感じ。HとTを比べる。
# S-278   <unk> 7 年 1 月 3 日 （ <unk> 年 2 月 20 日 ） - 即位
# T-278   The enthronement was held on February <<unk>> , <<unk>> .
# H-278   -3.647409677505493      In after , he Temple Temple in Kyoto , Kyoto City , and .
# D-278   -3.647409677505493      In after , he Temple Temple in Kyoto , Kyoto City , and .
# P-278   -3.7646 -4.4794 -2.5962 -2.4628 -4.6184 -4.3608 -3.3943 -3.8971 -3.3301 -4.7163 -4.6544 -3.3074 -3.9739 -4.8618 -0.2936
