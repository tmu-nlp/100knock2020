"""
# prepare pretrained model
!git clone https://github.com/MorinoseiMorizo/jparacrawl-finetune.git
!jparacrawl-finetune/ja-en/get-model.sh

# preprocess (w/o threshold)
!mkdir data-bin/kftt-wo-threshold.ja-en
!TEXT=jparacrawl-finetune/corpus/kftt-data-1.0/data/tok
!DICT=jparacrawl-finetune/pretrained_model_jaen
!fairseq-preprocess --source-lang ja --target-lang en \
    --trainpref $TEXT/kyoto-train \
    --validpref $TEXT/kyoto-dev \
    --testpref $TEXT/kyoto-test \
    --srcdict $DICT/dict.ja.txt \
    --tgtdict $DICT/dict.en.txt \
    --destdir data-bin/kftt-wo-threshold.ja-en/ \
    --workers 20

# fine tuning
!mkdir checkpoints/jesc-kftt.ja-en
!CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/kftt.ja-en/ \
    --reset-optimizer \
    --restore-file pretrained_model_jaen/base.pretrain.pt \
    --save-dir checkpoints/jesc-kftt.ja-en/ \
    --arch transformer --share-decoder-input-output-embed \
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

# prediction
!fairseq-interactive data-bin/kftt.ja-en/ \
    --path checkpoints/jesc-kftt.ja-en/checkpoint_best.pt \
    < ../kftt-data-1.0/data/tok/kyoto-test.ja \
    | grep '^H' | cut -f3 > 98.out

# evaluation
!fairseq-score --sys 98.out --ref ../kftt-data-1.0/data/tok/kyoto-test.en
BLEU4 = 5.77, 37.2/10.2/3.3/1.2 (BP=0.935, ratio=0.937, syslen=25053, reflen=26734)
"""
