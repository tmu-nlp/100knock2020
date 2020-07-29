#!/bin/zsh -eu
DATA=./data/bin/95_subwords
DROPOUTS=(0.1 0.2 0.3)
for N in `seq 3` ;
do
    DROPOUT=$DROPOUTS[$N]
    OUT=./checkpoints/97_subwords_dropout$DROPOUT
    fairseq-train $DATA \
        --seed 1 \
        --optimizer adam --clip-norm 0.0 \
        --arch transformer \
        --adam-betas '(0.9, 0.98)' \
        --share-decoder-input-output-embed \
        --dropout $DROPOUT \
        --warmup-init-lr 1e-07 \
        --min-lr 1e-09 \
        --update-freq 8 \
        --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 2000 \
        --weight-decay 0.0 \
        --max-tokens 4096 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --max-epoch 10 \
        --save-dir $OUT
done