"""
69. t-SNEによる可視化
ベクトル空間上の国名に関する単語ベクトルをt-SNEで可視化せよ．

[Ref]
- TSNE
    - https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

[MEMO]
2015 年版の knock99 に対応
"""
import os
import sys
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.pickle import load  # noqa: E402 isort:skip


if __name__ == "__main__":
    embeddings, country_names = load("chap07-embeddings-country")
    kmeans = load("chap07-kmeans")

    t_sne = TSNE(n_components=2).fit_transform(embeddings)

    plt.figure(figsize=(10, 5))
    cmap = plt.get_cmap("Set1")
    for name, coord, class_ in zip(country_names, t_sne, kmeans.labels_):
        cval = cmap(class_)
        plt.scatter(*coord, marker=".", color=cval, s=3)
        plt.annotate(name, xy=coord, color=cval, size=8)
    plt.savefig("out69.png")
