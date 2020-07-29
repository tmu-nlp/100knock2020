#!/bin/bash -eu

# 92. 機械翻訳モデルの適用
# 91 で学習したニューラル機械翻訳モデルを用い，与えられた（任意の）日本語の文を英語に翻訳するプログラムを実装せよ．
# 参考 https://github.com/pytorch/fairseq/tree/master/examples/translation

# 以下をかけば、動く。

CUDA_VISIBLE_DEVICES=0 fairseq-interactive data-bin/kftt.ja-en \
    --path knock91/checkpoint_best.pt

# プログラムを作れということなので、微妙。