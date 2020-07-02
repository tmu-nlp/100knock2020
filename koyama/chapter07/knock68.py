# 68. Ward法によるクラスタリング
# 国名に関する単語ベクトルに対し，Ward法による階層型クラスタリングを実行せよ．
# さらに，クラスタリング結果をデンドログラムとして可視化せよ．

from scipy.cluster.hierarchy import linkage, dendrogram
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # データを読み込む
    model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

    # 以下のWebページにある国名を使う
    # http://www.fao.org/countryprofiles/iso3list/en/
    countries = pd.read_table("countries.tsv")
    countries = countries["Short name"].values

    # 国名のベクトルを取り出す
    country_vec = []
    country_name = []
    for country in countries:
        if country in model.vocab:
            country_vec.append(model[country])
            country_name.append(country)

    # Ward法による階層型クラスタリング
    linkage_result = linkage(country_vec, method='ward', metric='euclidean')

    # デンドログラムで結果を表示する
    plt.figure(figsize=(16, 9))
    dendrogram(linkage_result, labels=country_name)
    plt.show()