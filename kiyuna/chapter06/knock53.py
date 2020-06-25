"""
53. 予測
52で学習したロジスティック回帰モデルを用い，
与えられた記事見出しからカテゴリとその予測確率を計算するプログラムを実装せよ．

[MEMO]
2015 年版の knock73 に対応
"""
import os
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer  # noqa: E402 isort:skip
from kiyuna.utils.pickle import dump, load  # noqa: E402 isort:skip

classes = load("chap06-encoder-classes")
categories = {
    "b": "business",
    "t": "science and technology",
    "e": "entertainment",
    "m": "health",
}


def vectorize(labels, titles):

    encoder = LabelEncoder()
    encoder.fit(load("chap06-encoder-classes"))
    labels_ = encoder.transform(labels)

    vectorizer = TfidfVectorizer(vocabulary=load("chap06-vectorizer-vocabs"))
    vectorizer.fit(titles)
    features_ = vectorizer.transform(titles)

    return features_, labels_


def load_dataset(path):
    labels, titles = zip(*(line.strip().split("\t") for line in open(path)))
    return vectorize(labels, titles)


def decoder(label):
    return categories[classes[label]]


if __name__ == "__main__":
    features, _ = load_dataset("./test.feature.txt")
    classifier = load("chap06-classifier")
    predicts = classifier.predict(features)
    probas = classifier.predict_proba(features)
    for predict, proba in zip(predicts, probas):
        print(f"{decoder(predict):23}\t{max(proba):f}")
