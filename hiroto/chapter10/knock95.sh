#!/bin/zsh -eu
OUTPUTS=./data/outputs/95
for N in `seq 20` ;
do
    fairseq-interactive \
	    --path ./checkpoints/95_subwords/checkpoint_best.pt \
        --beam $N \
       	./data/bin/95_subwords \
	    < ./data/subwords/test.ja | grep '^H' | cut -f3 > $OUTPUTS/beam$N.out
    
    python 95mold.py $OUTPUTS beam$N.out

    fairseq-score --sys $OUTPUTS/tok.beam$N.out --ref ./data/tok/test.en >> ./data/outputs/95scores.out
done

