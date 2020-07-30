#!/bin/zsh -eu
OUTPUTS=./data/outputs
fairseq-score --sys $OUTPUTS/91_sub.out --ref ./data/tok/test.en

OUTPUTS=./data/outputs
DEST=./results/97.result
DROPOUTS=(0.1 0.2 0.3)
for N in `seq 3` ;
do
    fairseq-score \
    --sys $OUTPUTS/97_subwords_dropout$DROPOUTS[$N].out \
    --ref ./data/subwords/test.en \
    >> $DEST
done

OUTPUTS=./data/outputs
fairseq-score --sys $OUTPUTS/98_subwords.out --ref ./data/jsec/subwords/test.en