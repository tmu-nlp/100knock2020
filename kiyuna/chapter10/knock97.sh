#!/bin/bash
set -eu
<<__EOF__
97. ハイパー・パラメータの調整
ニューラルネットワークのモデルや，そのハイパーパラメータを変更しつつ，
開発データにおけるBLEUスコアが最大となるモデルとハイパーパラメータを求めよ．

[Usage]
sh knock97.sh 1 KFTT.bpe.ja-en 10
__EOF__

GPU=$1
EXP_NAME=$2
MAX_EPOCH=${3:-2}
ARCH=transformer

knock91.bpe
rm -rf $EXP_NAME.tfm.log checkpoints/$EXP_NAME.tfm tensorboard/$EXP_NAME.tfm

CUDA_VISIBLE_DEVICES=$GPU fairseq-train data-bin/$EXP_NAME \
    --save-dir checkpoints/$EXP_NAME.tfm --seed 2020 --max-epoch $MAX_EPOCH \
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
    --tensorboard-logdir tensorboard/$EXP_NAME.tfm | tee $EXP_NAME.tfm.log

# knock92.bpe
mkdir -p dumped

SRC_PATH=data/kftt-data-1.0/data/tok/kyoto-test.ja
BPE_CODE=data/kftt-data-1.0/data/bpe/code

CUDA_VISIBLE_DEVICES=$GPU fairseq-interactive \
    --path checkpoints/$EXP_NAME.tfm/checkpoint_best.pt \
    --remove-bpe --bpe=subword_nmt --bpe-codes $BPE_CODE \
    data-bin/$EXP_NAME <$SRC_PATH | grep '^H' | cut -f3 >dumped/k92.$EXP_NAME.tfm.txt

# knock93.bpe
sh knock93.sh $EXP_NAME.tfm
