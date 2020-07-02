"""
57. 特徴量の重みの確認
52で学習したロジスティック回帰モデルの中で，
重みの高い特徴量トップ10と，重みの低い特徴量トップ10を確認せよ．

[MEMO]
2015 年版の knock75 に対応
"""
import os
import sys

from sklearn.metrics import precision_recall_fscore_support

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import message, Renderer  # noqa: E402 isort:skip
from kiyuna.utils.pickle import dump, load  # noqa: E402 isort:skip


if __name__ == "__main__":
    classifier = load("chap06-classifier")
    names = load("chap06-vectorizer-names")
    weights = classifier.coef_.flatten()
    ranking = sorted(zip(weights, names), reverse=True)
    with Renderer("knock57") as out:
        out.header("best 10")
        for weight, name in ranking[:10]:
            message(f"{name:15}{weight:f}")
        out.header("worst 10")
        for weight, name in ranking[:-11:-1]:
            message(f"{name:15}{weight:f}")
