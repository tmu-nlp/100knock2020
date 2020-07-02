#国名に関する単語ベクトルを抽出し，
#k-meansクラスタリングをクラスタ数k=5として実行せよ．

from sklearn.cluster import KMeans
import numpy as np
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin", binary=True
)

countrydic = []
countries_vec = []
with open("countries_v1.txt", "r") as f2:
  countries = f2.readlines()
for line in countries:
  line = line.strip()
  try:
    countries_vec.append(model[line])
  except KeyError:
    try:
      line = line.replace(" ", "_")
      countries_vec.append(model[line])
    except KeyError:
      print(line)

pred = KMeans(n_clusters=5, random_state=0)
print(pred.fit_predict(np.array(countries_vec)))

"""
array([2, 1, 2, 0, 3, 4, 2, 2, 3, 3, 2, 1, 1, 4, 2, 4, 0, 1, 4, 0, 4, 1,
       2, 0, 0, 1, 0, 3, 0, 1, 4, 1, 4, 0, 4, 3, 4, 3, 3, 0, 1, 2, 0, 3,
       3, 0, 2, 3, 0, 2, 3, 3, 1, 1, 1, 1, 3, 1, 3, 4, 1, 1, 2, 0, 1, 1,
       2, 2, 2, 1, 0, 2, 3, 0, 1, 1, 0, 4, 2, 3, 1, 3, 3, 1, 1, 4, 1, 4,
       4, 3, 4, 1, 2, 1, 3, 3, 0, 3, 1, 4, 3, 3, 1, 0, 1, 2, 0, 2, 4, 2,
       1, 3, 3, 4, 1, 0, 0], dtype=int32)
"""