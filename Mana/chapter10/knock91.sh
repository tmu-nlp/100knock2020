#!/bin/bash -eu
# 91. 機械翻訳モデルの訓練
# 90で準備したデータを用いて，ニューラル機械翻訳のモデルを学習せよ（ニューラルネットワークのモデルはTransformerやLSTMなど適当に選んでよい）．

# fp16 をすると訓練時間が短縮できる？
# max-epochを指定しないとなかなか終わらない。
# 参考 https://github.com/pytorch/fairseq/tree/master/examples/translation

CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/kftt.ja-en \
    --fp16 \
    --save-dir knock91 --max-epoch 10 --arch lstm --share-decoder-input-output-embed \
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
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric > knock91.log