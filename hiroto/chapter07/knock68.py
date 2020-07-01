'''
68. Ward法によるクラスタリングPermalink
国名に関する単語ベクトルに対し，Ward法による階層型クラスタリングを実行せよ．
さらに，クラスタリング結果をデンドログラムとして可視化せよ．
'''
import pickle
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

with open('./data/country_names.pickle', mode='rb') as f1\
    , open('./data/country_vectors.pickle', mode='rb') as f2:
    country_names = pickle.load(f1)
    country_vectors = pickle.load(f2)

Z = linkage(country_vectors, method='ward')
plt.figure(figsize=(16, 9))
dendrogram(Z, labels=country_names, leaf_font_size=7)
plt.show()
