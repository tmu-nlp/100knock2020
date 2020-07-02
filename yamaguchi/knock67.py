# 学習済み単語ベクトルを扱うためにインポート
import gensim
# k-meansクラスタリングをするためにインポート
from sklearn.cluster import KMeans

import numpy as np

def main():
    # 「word2vec」で単語をベクトル化する
    # 「save_word2vec_format」で保存したモデルを「load_word2vec_format」で読み込む
    # 「KeyedVectors」で追加学習に必要なデータを除いてWord2Vecのモデルを軽量化
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    # 「United States」は内部的に「United_States」と表現されている
    model['United_States']

    # 国名の取得
    countries = set()
    with open('knock67.txt') as f:
        for line in f:
            line = line.split()
            if line[0] in ['capital-common-countries', 'capital-world']:
                countries.add(line[2])
            elif line[0] in ['currency', 'gram6-nationality-adjective']:
                countries.add(line[1])
    countries = list(countries)

    # 単語ベクトルの取得
    countries_vec = [model[country] for country in countries]

    from sklearn.cluster import KMeans

    # k-meansクラスタリング
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(countries_vec)
    for i in range(5):
        cluster = np.where(kmeans.labels_ == i)[0]
        print('cluster', i)
        print(', '.join([countries[k] for k in cluster]))

if __name__ == '__main__':
    main()
