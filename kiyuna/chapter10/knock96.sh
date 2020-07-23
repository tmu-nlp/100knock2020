#!/bin/bash
set -eu
<< __EOF__
96. 学習過程の可視化
Tensorboardなどのツールを用い，ニューラル機械翻訳モデルが学習されていく過程を可視化せよ．
可視化する項目としては，学習データにおける損失関数の値とBLEUスコア，
開発データにおける損失関数の値とBLEUスコアなどを採用せよ．

[Usage]
sh knock96.sh tensorboard/$EXP_NAME
__EOF__

tensorboard --logdir $1
