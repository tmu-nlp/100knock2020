#ベクトル空間上の国名に関する単語ベクトルをt-SNEで可視化せよ．

#国名に関する単語ベクトルに対し，Ward法による階層型クラスタリングを実行せよ．
#さらに，クラスタリング結果をデンドログラムとして可視化せよ．
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE

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
tsne = TSNE(random_state=0, n_iter=15000, metric='cosine')
embs = tsne.fit_transform(X)
plt.figure(figsize=(16, 12))
for i in range(len(embs)):
  plt.scatter(embs[i][0], embs[i][1])
  plt.annotate(country_name[i], (embs[i][0], embs[i][1]))
plt.savefig('tsne1.png')
plt.show()