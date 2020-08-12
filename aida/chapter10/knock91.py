"""
!pip install sacremoses

!mkdir checkpoints
!mkdir checkpoints/kftt.ja-en

!CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/kftt.ja-en \
    --save-dir checkpoints/kftt.ja-en/ \
    --arch lstm --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm     0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4    000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothi    ng 0.1 \
    --max-tokens 4096 \
    --max-epoch 5 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len    _b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

"""

