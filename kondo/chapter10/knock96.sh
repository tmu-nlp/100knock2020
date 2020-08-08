"""
96. 学習過程の可視化Permalink
Tensorboardなどのツールを用い，ニューラル機械翻訳モデルが学習されていく過程を可視化せよ．
可視化する項目としては，学習データにおける損失関数の値とBLEUスコア，開発データにおける損失関数の値とBLEUスコアなどを採用せよ．
"""

#!/bin/bash
OUT_DIR=model/knock96
mkdir -p $OUT_DIR

CUDA_VISIBLE_DEVICES=5 PYTHONIOENCODING=utf-8 nohup fairseq-train \
    ./use_file/AT_bin \
    --tensorboard-logdir log96 \
    --save-dir $OUT_DIR \
    --arch lstm --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 1024 \
    --max-epoch 10 \
    --maximize-best-checkpoint-metric > log/train96.log 