"""
55. 混同行列の作成
52で学習したロジスティック回帰モデルの混同行列（confusion matrix）を，
学習データおよび評価データ上で作成せよ．

[MEMO]
2015 年版の knock77 に対応
"""
import os
import sys

from sklearn.metrics import confusion_matrix

from knock53 import load_dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer  # noqa: E402 isort:skip
from kiyuna.utils.pickle import dump, load  # noqa: E402 isort:skip


if __name__ == "__main__":
    classifier = load("chap06-classifier")
    with Renderer("knock55") as out:
        for name in "train", "test":
            features, labels = load_dataset(f"./{name}.feature.txt")
            predicts = classifier.predict(features)
            out.result(name, confusion_matrix(labels, predicts))
