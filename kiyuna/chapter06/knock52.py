"""
52. 学習
51で構築した学習データを用いて，ロジスティック回帰モデルを学習せよ．

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


def vectorize_init(labels, titles):

    encoder = LabelEncoder()
    encoder.fit(labels)
    labels_ = encoder.transform(labels)
    dump(encoder.classes_, "chap06-encoder-classes")  # <class 'numpy.ndarray'>

    vectorizer = TfidfVectorizer()
    vectorizer.fit(titles)
    features_ = vectorizer.transform(titles)
    dump(vectorizer.vocabulary_, "chap06-vectorizer-vocabs")  # <class 'dict'>
    dump(vectorizer.get_feature_names(), "chap06-vectorizer-names")  # [+] saved : chap06-model

    return features_, labels_


def load_dataset(path):
    labels, titles = zip(*(line.strip().split("\t") for line in open(path)))
    return vectorize_init(labels, titles)


if __name__ == "__main__":
    features, labels = load_dataset("./train.feature.txt")
    classifier = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=123)
    classifier.fit(features, labels)
    dump(classifier, "chap06-classifier")
