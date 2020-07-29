#!/bin/zsh -eu
DATA=./data/bin/98_subwords/
OUT=./checkpoints/98_subwords
fairseq-train $DATA \
    --seed 1 \
    --optimizer adam --clip-norm 0.0 \
    --arch transformer \
    --adam-betas '(0.9, 0.98)' \
    --share-decoder-input-output-embed \
    --dropout 0.1 \
    --warmup-init-lr 1e-07 \
    --min-lr 1e-09 \
    --update-freq 8 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --weight-decay 0.0 \
    --max-tokens 4096 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-update 50000 \
    --keep-last-epochs 10 \
    --save-dir $OUT