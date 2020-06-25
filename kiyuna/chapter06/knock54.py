"""
54. 正解率の計測
52で学習したロジスティック回帰モデルの正解率を，学習データおよび評価データ上で計測せよ．

[MEMO]
2015 年版の knock77 に対応
"""
import os
import sys

from knock53 import load_dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer  # noqa: E402 isort:skip
from kiyuna.utils.pickle import dump, load  # noqa: E402 isort:skip


if __name__ == "__main__":
    classifier = load("chap06-classifier")
    with Renderer("knock54") as out:
        for name in "train", "test":
            score = classifier.score(*load_dataset(f"./{name}.feature.txt"))
            out.result(name, score)
