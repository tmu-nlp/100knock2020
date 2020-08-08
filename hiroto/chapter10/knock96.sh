OUT=./checkpoints/96_subwords
#rm -rf $OUT/*
#--warmup-init-lr 1e-7 \
#--adam-betas '(0.9,0.98)'
fairseq-train $DATA \
    --optimizer adam --clip-norm 1.0 \
    --arch transformer \
    --share-decoder-input-output-embed \
    --dropout 0.1 \
    --lr 1e-5 --lr-scheduler inverse_sqrt --warmup-updates 1000 \
    --weight-decay 0.0 \
    --max-tokens 8192 \
    --update-freq 1 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-epoch 10 \
    --save-dir $OUT \
    --tensorboard-logdir ./log/96tensorboard