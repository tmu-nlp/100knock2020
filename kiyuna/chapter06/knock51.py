"""
51. 特徴量抽出
学習データ，検証データ，評価データから特徴量を抽出し，それぞれtrain.feature.txt，
valid.feature.txt，test.feature.txtというファイル名で保存せよ．
なお，カテゴリ分類に有用そうな特徴量は各自で自由に設計せよ．
記事の見出しを単語列に変換したものが最低限のベースラインとなるであろう．

[MEMO]
2015 年版の knock71-72 に対応
"""
import os
import sys

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer as PS

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer  # noqa: E402 isort:skip


stop_words = set(stopwords.words("english"))


def check(stem):
    if stem in stop_words:
        return False
    if len(stem) == 1:
        return False
    return True


def extract_features(title):
    tokens = title.split(" ")
    tokens = filter(check, map(PS().stem, tokens))
    return " ".join(tokens)


def func(name):
    with open(name + ".txt") as f_in, open(name + ".feature.txt", "w") as f_out:
        for line in f_in:
            category, title = line.strip().split("\t")
            print(category, extract_features(title), sep="\t", file=f_out)


for dataset in "train", "valid", "test":
    func(dataset)
