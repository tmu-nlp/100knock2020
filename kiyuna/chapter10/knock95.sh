#!/bin/bash
set -eu
<< __EOF__
95. サブワード化
トークンの単位を単語や形態素からサブワードに変更し，91-94の実験を再度実施せよ．

[Usage]
sh knock95.sh 4
__EOF__

GPU=$1
EXP_NAME=KFTT.bpe.ja-en
MAX_EPOCH=10

bash knock90.bpe.sh
sh knock91.bpe.sh $GPU $EXP_NAME $MAX_EPOCH
sh knock92.bpe.sh $GPU $EXP_NAME
sh knock93.sh          $EXP_NAME
sh knock94.bpe.sh $GPU $EXP_NAME
