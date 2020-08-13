fairseq-train data-en-de-bpe-prep95 \
    --fp16 \
    --save-dir save96-lstm-en-de \
    --max-epoch 10 \
    --lr 1e-3 \
    --optimizer adam --clip-norm 0.1 \
    --dropout 0.2 \
    --max-tokens 4000 \
    --arch lstm \
    --tensorboard-logdir log-en-de-96