"""
# train models (lr changed)
!mkdir checkpoints/kftt-search.ja-en/

!mkdir checkpoints/kftt-search.ja-en/lr-5e-3
!CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/kftt.ja-en \
    --save-dir checkpoints/kftt-search.ja-en/lr-5e-3/ \
    --arch lstm --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --max-epoch 5 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

!mkdir checkpoints/kftt-search.ja-en/lr-5e-5/
!CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/kftt.ja-en \
    --save-dir checkpoints/kftt-search.ja-en/lr-5e-5/ \
    --arch lstm --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-5 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --max-epoch 5 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

# translation
!fairseq-interactive data-bin/kftt.ja-en/ \
    --path checkpoints/kftt-search.ja-en/lr-5e-3/checkpoint_best.pt \
    < ../kftt-data-1.0/data/tok/kyoto-test.ja \
    | grep '^H' | cut -f3 > 97.3.out

!fairseq-interactive data-bin/kftt.ja-en/ \
    --path checkpoints/kftt-search.ja-en/lr-5e-5/checkpoint_best.pt \
    < ../kftt-data-1.0/data/tok/kyoto-test.ja \
    | grep '^H' | cut -f3 > 97.5.out

# evaluation
## lr=5e-3
!fairseq-score --sys 97.3.out --ref ../kftt-data-1.0/data/tok/kyoto-test.en
BLEU4 = 16.24, 46.1/21.3/11.7/6.9 (BP=0.967, ratio=0.967, syslen=25862, reflen=26734)

## lr=5e-4
!fairseq-score --sys 92.out --ref ../kftt-data-1.0/data/tok/kyoto-test.en
BLEU4 = 14.61, 43.0/18.6/10.0/5.7 (BP=1.000, ratio=1.019, syslen=27238, reflen=26734)

## lr=5e-5
!fairseq-score --sys 97.5.out --ref ../kftt-data-1.0/data/tok/kyoto-test.en
BLEU4 = 2.27, 21.0/4.1/0.9/0.3 (BP=1.000, ratio=1.239, syslen=33127, reflen=26734)

"""

