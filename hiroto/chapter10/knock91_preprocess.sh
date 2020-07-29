#!/bin/zsh -eu
TEXT=./data/tok
DEST=./data/bin/91_sub
fairseq-preprocess -s ja -t en \
    --trainpref $TEXT/train \
    --validpref $TEXT/dev \
    --testpref $TEXT/test \
    --thresholdsrc 20 \
    --thresholdtgt 20 \
    --destdir $DEST
