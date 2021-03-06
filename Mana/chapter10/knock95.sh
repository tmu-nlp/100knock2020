#!/bin/bash -eu

# 95. サブワード化
# トークンの単位を単語や形態素からサブワードに変更し，91-94の実験を再度実施せよ．
# 参考 https://fairseq.readthedocs.io/en/latest/command_line_tools.html

# --bpeを指定する。例えばsubword_nmtなど。

CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/kftt.ja-en \
    --fp16 \
    --save-dir knock95 --bpe subword_nmt　--max-epoch 10 --arch lstm --share-decoder-input-output-embed \
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
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric > knock95.log