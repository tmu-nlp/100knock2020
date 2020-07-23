#!/bin/bash
set -eu

GPU=$1

SRC=data/kftt-data-1.0/data/tok
DST=data/kftt-data-1.0/data/mini

mkdir -p $DST
for file in `find $SRC -type f`; do
    echo `basename $file`
    head $file -n 300 > $DST/`basename $file`
done

EXP_NAME=KFTT.mini.ja-en
rm -rf data-bin/$EXP_NAME

# Preprocess/binarize the data
fairseq-preprocess --source-lang ja --target-lang en \
    --trainpref $DST/kyoto-train \
    --validpref $DST/kyoto-dev \
    --testpref $DST/kyoto-test \
    --destdir data-bin/$EXP_NAME \
    --workers 20

sh knock91.sh 4 $EXP_NAME
