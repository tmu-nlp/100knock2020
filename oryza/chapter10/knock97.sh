fairseq-train data-en-de-bpe-prep95 \
    --fp16 \
    --save-dir save97-trans-en-de \
    --max-epoch 50 \
    --lr 1e-3 \
    --optimizer adam --clip-norm 0.1 \
    --dropout 0.2 \
    --max-tokens 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --arch transformer