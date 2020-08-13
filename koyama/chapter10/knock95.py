# 95. サブワード化
# トークンの単位を単語や形態素からサブワードに変更し，
# 91-94の実験を再度実施せよ．


# 以下はBPEをかけるスクリプト
# BPEをかけた後は，91-94まで同じことをやる
'''
#!/bin/bash

set -x
set -e


SCRIPT_DIR=$(cd $(dirname $0); pwd)
DATA_DIR=$1
bpe_operations=16000

## paths to training and development datasets
src_ext=ja
trg_ext=en
train_data_prefix=$DATA_DIR/train
dev_data_prefix=$DATA_DIR/dev
test_data_prefix=$DATA_DIR/test


######################
# subword segmentation
PROCESSED=$DATA_DIR/processed_$bpe_operations
mkdir -p $PROCESSED/

cat $train_data_prefix.tok.$src_ext | subword-nmt learn-bpe -s $bpe_operations > $PROCESSED/train.bpe.model.ja
cat $train_data_prefix.tok.$trg_ext | subword-nmt learn-bpe -s $bpe_operations > $PROCESSED/train.bpe.model.en

subword-nmt apply-bpe -c $PROCESSED/train.bpe.model.ja < $train_data_prefix.tok.$src_ext > $PROCESSED/train.ja
subword-nmt apply-bpe -c $PROCESSED/train.bpe.model.en < $train_data_prefix.tok.$trg_ext > $PROCESSED/train.en
subword-nmt apply-bpe -c $PROCESSED/train.bpe.model.ja < $dev_data_prefix.tok.$src_ext > $PROCESSED/dev.ja
subword-nmt apply-bpe -c $PROCESSED/train.bpe.model.en < $dev_data_prefix.tok.$trg_ext > $PROCESSED/dev.en
subword-nmt apply-bpe -c $PROCESSED/train.bpe.model.ja < $test_data_prefix.tok.$src_ext > $PROCESSED/test.ja
subword-nmt apply-bpe -c $PROCESSED/train.bpe.model.en < $test_data_prefix.tok.$trg_ext > $PROCESSED/test.en
'''