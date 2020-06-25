"""
57. 特徴量の重みの確認Permalink
52で学習したロジスティック回帰モデルの中で，重みの高い特徴量トップ10と，
重みの低い特徴量トップ10を確認せよ．
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from knock53 import score_lg, logistic_model, X_train

if __name__ == "__main__":
    #バイナリファイルとして保存したモデルをロードする
    with open(logistic_model, "rb") as f:
        lg = pickle.load(f)
    
    features = X_train.columns
    for cla, coef in zip(lg.classes_, lg.coef_):
        print("CATEGORY: {}".format(cla))
        coef = features[np.argsort(coef)] #index(ここでは何番目か)を返す featureはそれに対応する単語
        best10 = pd.DataFrame(coef[:-11:-1], columns = ["top10"], index = range(1, 11))  #後ろから10個
        worst10 = pd.DataFrame(coef[:10], columns = ["worst10"], index = range(1, 11))  #前から10個
        weights = pd.concat([best10, worst10], axis=1)
        print(weights)
        print()

"""
CATEGORY: b
        top10    worst10
1        bank      video
2         fed      ebola
3       china        her
4         ecb        the
5      stocks        and
6        euro        she
7   obamacare     google
8         oil      apple
9      yellen      virus
10     dollar  microsoft

CATEGORY: e
         top10   worst10
1   kardashian        us
2        chris    update
3          her    google
4        movie     study
5         star        gm
6         film     china
7         paul       ceo
8           he  facebook
9          she      says
10     wedding     apple

CATEGORY: m
         top10   worst10
1        ebola  facebook
2        study        gm
3       cancer       ceo
4         drug     apple
5         mers      bank
6          fda      deal
7        cases     sales
8   cigarettes    google
9          cdc   climate
10       could   twitter

CATEGORY: t
         top10   worst10
1       google    stocks
2     facebook       fed
3        apple       her
4    microsoft   percent
5      climate      drug
6           gm  american
7         nasa   ukraine
8        tesla    cancer
9      comcast     still
10  heartbleed    shares
"""