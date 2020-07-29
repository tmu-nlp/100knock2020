#!/bin/bash -eu
# 90. データの準備
# 機械翻訳のデータセットをダウンロードせよ．訓練データ，開発データ，評価データを整形し，必要に応じてトークン化などの前処理を行うこと．ただし，この段階ではトークンの単位として形態素（日本語）および単語（英語）を採用せよ．
# 参考 https://github.com/pytorch/fairseq/tree/master/examples/translation

# download KFTT
wget http://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz
tar xzvf kftt-data-1.0.tar.gz
rm kftt-data-1.0.tar.gz

# preprocess
TEXT=kftt-data-1.0/data/tok
fairseq-preprocess --source-lang ja --target-lang en \
    --trainpref $TEXT/kyoto-train --validpref $TEXT/kyoto-dev --testpref $TEXT/kyoto-test \
    --destdir data-bin/kftt.ja-en \
    --workers 20