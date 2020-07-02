"""
68. Ward法によるクラスタリング
国名に関する単語ベクトルに対し，Ward法による階層型クラスタリングを実行せよ．
さらに，クラスタリング結果をデンドログラムとして可視化せよ．

[Ref]
- linkage
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
- dendrogram
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html

[MEMO]
2015 年版の knock98 に対応
"""
import os
import sys
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.pickle import load  # noqa: E402 isort:skip

if __name__ == "__main__":
    embeddings, country_names = load("chap07-embeddings-country")

    # Ward 法による階層型クラスタリング
    z = linkage(embeddings, method="ward")

    # クラスタリング結果をデンドログラムとして可視化
    plt.figure(figsize=(20, 10))
    dendrogram(z, labels=country_names, leaf_font_size=10)
    plt.savefig("out68.png")
