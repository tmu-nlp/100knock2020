#!/bin/bash -eu

# 96. 学習過程の可視化
# Tensorboardなどのツールを用い，ニューラル機械翻訳モデルが学習されていく過程を可視化せよ．可視化する項目としては，学習データにおける損失関数の値とBLEUスコア，開発データにおける損失関数の値とBLEUスコアなどを採用せよ．

CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/kftt.ja-en \
    --fp16 \
    --save-dir knock96 --bpe subword_nmt　--max-epoch 10 --arch lstm --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --tensorboard-logdir knock96.log

# pip install tensorflow tensorboardX
# tensorboard --logdir knock96.log --bind_all
# 示されたサーバーにアクセス。
# 研究室からだと--bind_allが必要。