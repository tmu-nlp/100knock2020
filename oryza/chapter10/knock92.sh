fairseq-interactive --path save91-lstm-en-de/checkpoint_best.pt data-en-de-prep91 < test.de-en.en | grep '^H' | cut -f3 > knock92.out