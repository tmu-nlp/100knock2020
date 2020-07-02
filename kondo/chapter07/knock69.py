"""
69. t-SNEによる可視化Permalink
ベクトル空間上の国名に関する単語ベクトルをt-SNEで可視化せよ．
"""

from sklearn.manifold import TSNE
from knock67 import make_dataframe, collect_countries
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataframe = make_dataframe(collect_countries())
    t_sne = TSNE().fit_transform(dataframe.iloc[:, 1:])
    #print(t_sne)
    df_t_sne = pd.DataFrame(t_sne)
    df_t_sne = pd.concat([df_t_sne, dataframe.iloc[:, 0]], axis=1, ignore_index=True)
    a = df_t_sne.plot.scatter(x=0, y=1, s=5)
    for k, v in df_t_sne.iterrows():
        a.annotate(v[2], xy=(v[0], v[1]), size=5)
    plt.show()