#!/bin/bash
set -eu
<<__EOF__
91. 機械翻訳モデルの訓練
90で準備したデータを用いて，ニューラル機械翻訳のモデルを学習せよ
（ニューラルネットワークのモデルはTransformerやLSTMなど適当に選んでよい）．

[Usage]
sh knock91.bpe.sh 5 KFTT.bpe.ja-en 10
__EOF__

GPU=$1
EXP_NAME=$2
MAX_EPOCH=${3:-2}
ARCH=lstm

rm -rf $EXP_NAME.log checkpoints/$EXP_NAME tensorboard/$EXP_NAME

CUDA_VISIBLE_DEVICES=$GPU fairseq-train data-bin/$EXP_NAME \
    --save-dir checkpoints/$EXP_NAME --seed 2020 --max-epoch $MAX_EPOCH \
    --bpe subword_nmt \
    --arch $ARCH --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --tensorboard-logdir tensorboard/$EXP_NAME | tee $EXP_NAME.log
