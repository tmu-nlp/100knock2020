#国名に関する単語ベクトルに対し，Ward法による階層型クラスタリングを実行せよ．
#さらに，クラスタリング結果をデンドログラムとして可視化せよ．
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin", binary=True
)

country_name = []
countries_vec = []
with open("countries_v1.txt", "r") as f2:
  countries = f2.readlines()
for line in countries:
  line = line.strip()
  try:
    country_name.append(line)
    countries_vec.append(model[line])
  except KeyError:
    try:
      line = line.replace(" ", "_")
      country_name.append(line)
      countries_vec.append(model[line])
    except KeyError:
      #print(line)

X = np.array(countries_vec)
linkage_result = linkage(X, method='ward', metric='euclidean')
plt.figure(num=None, figsize=(16, 9), dpi=200, facecolor='w', edgecolor='k')
dendrogram(linkage_result, labels=country_name)
plt.show()