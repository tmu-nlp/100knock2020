#!/bin/bash
OUT_DIR=model/model1_m
mkdir -p $OUT_DIR

CUDA_VISIBLE_DEVICES=1 nohup fairseq-train \
    ./use_file/AT_bin \
    --save-dir $OUT_DIR \
    --arch lstm --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 1024 \
    --max-epoch 10 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric > log/train.log 