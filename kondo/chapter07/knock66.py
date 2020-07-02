"""
66. WordSimilarity-353での評価Permalink
The WordSimilarity-353 Test Collectionの評価データをダウンロードし，
単語ベクトルにより計算される類似度のランキングと，
人間の類似度判定のランキングの間のスピアマン相関係数を計算せよ．
"""

import pandas as pd
import numpy as np
import pickle
import csv
from scipy.stats import spearmanr
from knock61 import cos_similarity

data_file = "set1.csv"

model_file = "model.sav"
with open(model_file, "rb") as file_model:
    model = pickle.load(file_model)

if __name__ == "__main__":
    human = []
    w2v = []
    with open(data_file) as f:
        data = csv.reader(f)
        for i, X in enumerate(data):
            if i == 0:
                continue
            sim = cos_similarity(model[X[0]], model[X[1]])
            human.append(sim)
            w2v.append(X[2])
        human = np.array(human)
        w2v = np.array(w2v)

        correlation, pvalue = spearmanr(human, w2v)
    print(f"{correlation}")

#0.6332629395311367