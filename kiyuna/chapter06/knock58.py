"""
58. 正則化パラメータの変更
ロジスティック回帰モデルを学習するとき，正則化パラメータを調整することで，
学習時の過学習（overfitting）の度合いを制御できる．
異なる正則化パラメータでロジスティック回帰モデルを学習し，
学習データ，検証データ，および評価データ上の正解率を求めよ．
実験の結果は，正則化パラメータを横軸，正解率を縦軸としたグラフにまとめよ．

[MEMO]
New
"""
import os
import sys

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from knock53 import load_dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import message, Renderer  # noqa: E402 isort:skip
from kiyuna.utils.pickle import dump, load  # noqa: E402 isort:skip

datasets = ["train", "valid", "test"]


def calc_accuracy(C):
    classifier = LogisticRegression(
        C=C, multi_class="multinomial", solver="lbfgs", random_state=123
    ).fit(*load_dataset(f"./train.feature.txt"))
    res = []
    for dataset in datasets:
        res.append(classifier.score(*data[dataset]))
    return res


if __name__ == "__main__":
    data = {}
    for dataset in datasets:
        data[dataset] = load_dataset(f"./{dataset}.feature.txt")

    Cs = [10 ** i for i in range(-4, 5)]

    plt.plot(Cs, tuple(map(calc_accuracy, Cs)))
    plt.xlabel("Inverse of regularization strength (C)")
    plt.ylabel("Accuracy")
    plt.xscale("log")
    plt.legend(datasets)
    plt.savefig("out58.png")
