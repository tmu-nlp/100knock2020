"""

!mkdir data-bin/kftt-bpe.ja-en
TEXT=../kftt-data-1.0/data/tok/

# define bpe model
!git clone https://github.com/rsennrich/subword-nmt.git
!mkdir ../kftt-data-1.0/data/bpe

SRC=../kftt-data-1.0/data/tok
DST=../kftt-data-1.0/data/bpe
BPE_TOKENS=10000
BPE_CODE=../kftt-data-1.0/data/bpe/code

!cat $SRC/kyoto-train.ja $SRC/kyoto-train.en > $SRC/kyoto-train.ja-en
!python ./subword-nmt/subword_nmt/learn_bpe.py -s $BPE_TOKENS < $SRC/kyoto-train.ja-en > $BPE_CODE

# adopt bpe model
python ./subword-nmt/subword_nmt/apply_bpe.py -c $BPE_CODE     < $SRC/kyoto-train.ja > $DST/kyoto-train.ja
!python ./subword-nmt/subword_nmt/apply_bpe.py -c $BPE_CODE     < $SRC/kyoto-train.en > $DST/kyoto-train.en
!python ./subword-nmt/subword_nmt/apply_bpe.py -c $BPE_CODE     < $SRC/kyoto-dev.ja > $DST/kyoto-dev.ja
!python ./subword-nmt/subword_nmt/apply_bpe.py -c $BPE_CODE     < $SRC/kyoto-dev.en > $DST/kyoto-dev.en
!python ./subword-nmt/subword_nmt/apply_bpe.py -c $BPE_CODE     < $SRC/kyoto-test.ja > $DST/kyoto-test.ja
!python ./subword-nmt/subword_nmt/apply_bpe.py -c $BPE_CODE     < $SRC/kyoto-test.en > $DST/kyoto-test.en

# preprocess
BPETEXT = '../kftt-data-1.0/data/bpe'
!fairseq-preprocess --source-lang ja --target-lang en \
    --trainpref $BPETEXT/kyoto-train \
    --validpref $BPETEXT/kyoto-dev \
    --testpref $BPETEXT/kyoto-test \
    --destdir data-bin/kftt-bpe.ja-en/ \
    --thresholdsrc 5 \
    --thresholdtgt 5 \
    --bpe subword_nmt \
    --workers 20

# training
!mkdir checkpoints/kftt-bpe.ja-en
!CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/kftt-bpe.ja-en \
    --save-dir checkpoints/kftt-bpe.ja-en/ \
    --arch lstm --share-decoder-input-output-embed \
    --bpe subword_nmt \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
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
!pip install subword-nmt
!fairseq-interactive data-bin/kftt-bpe.ja-en \
    --path checkpoints/kftt-bpe.ja-en/checkpoint_best.pt \
    --remove-bpe --bpe subword_nmt --bpe-codes $BPE_CODE\
    < ../kftt-data-1.0/data/tok/kyoto-test.ja \
    | grep '^H' | cut -f3 > 95.out

# evaluation
!fairseq-score --sys 95.out --ref ../kftt-data-1.0/data/tok/kyoto-test.en
BLEU4 = 14.40, 41.7/18.1/9.8/5.8 (BP=1.000, ratio=1.084, syslen=28988, reflen=26734)
"""

