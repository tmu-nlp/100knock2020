#!/bin/zsh -eu
OUTPUTS=./data/outputs/94
for N in `seq 100` ;
do
    fairseq-interactive \
	    --path ./checkpoints/91_sub/checkpoint_best.pt \
        --beam $N \
       	./data/bin/91 \
	    < ./data/tok/test.ja | grep '^H' | cut -f3 > $OUTPUTS/beam$N.out

    fairseq-score --sys $OUTPUTS/beam$N.out --ref ./data/tok/test.en >> ./data/outputs/94scores.out
done

