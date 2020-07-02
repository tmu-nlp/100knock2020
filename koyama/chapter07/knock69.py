# 69. t-SNEによる可視化
# ベクトル空間上の国名に関する単語ベクトルをt-SNEで可視化せよ．

from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
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

    # t-SNEで可視化する
    tsne = TSNE(random_state=0)
    embs = tsne.fit_transform(country_vec)
    plt.scatter(embs[:, 0], embs[:, 1])
    plt.show()