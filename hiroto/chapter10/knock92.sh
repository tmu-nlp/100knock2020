#!/bin/zsh -eu
fairseq-interactive \
	--path ./checkpoints/98_subwords/checkpoint_best.pt \
       	./data/bin/98_subwords \
	< ./data/jsec/subwords/test.ja | grep '^H' | cut -f3 > ./data/outputs/98_subwords.out
