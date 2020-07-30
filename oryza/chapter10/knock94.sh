for N in `seq 1 20`; do
    fairseq-interactive --path save91-lstm-en-de/checkpoint_best.pt --beam $N data-en-de-prep91 < test.de-en.en | grep '^H' | cut -f3 > knock94.$N.out
done

for N in `seq 1 20` ; do
    fairseq-score --sys knock94.$N.out --ref test.de-en.en > knock94.$N.score
done