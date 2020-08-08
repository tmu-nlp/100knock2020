#!/bin/bash
OUT_DIR=model/model98
mkdir -p $OUT_DIR

CUDA_VISIBLE_DEVICES=1 nohup fairseq-train \
    ./use_file_bpe/AT_bin \
    --distributed-world-size 1 \
    --save-dir $OUT_DIR \
    --arch transformer \
    --seed 1 \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --lr 0.0005 \
    --min-lr 1e-09 \
    --update-freq 8 \
    --dropout 0.1 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 1024 \
    --max-epoch 10 \
    --share-decoder-input-output-embed
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric > log/train98.log &
