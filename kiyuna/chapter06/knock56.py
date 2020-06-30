"""
56. 適合率，再現率，F1スコアの計測
52で学習したロジスティック回帰モデルの適合率，再現率，F1スコアを，評価データ上で計測せよ．
カテゴリごとに適合率，再現率，F1スコアを求め，
カテゴリごとの性能をマイクロ平均（micro-average）とマクロ平均（macro-average）で統合せよ．

[MEMO]
2015 年版の knock77 に対応
"""
import os
import sys

from sklearn.metrics import precision_recall_fscore_support

from knock53 import load_dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import message, Renderer  # noqa: E402 isort:skip
from kiyuna.utils.pickle import dump, load  # noqa: E402 isort:skip


if __name__ == "__main__":
    score_names = ["Precision", "Recall", "F1_score"]
    classifier = load("chap06-classifier")
    with Renderer("knock56") as out:
        for average in "micro", "macro":
            out.header(average)
            features, labels = load_dataset(f"./test.feature.txt")
            predicts = classifier.predict(features)
            for name, result in zip(
                score_names, precision_recall_fscore_support(labels, predicts, average=average)
            ):
                message(f"{name:10}\t{result}")
