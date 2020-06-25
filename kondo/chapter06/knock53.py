"""
53. 予測Permalink
52で学習したロジスティック回帰モデルを用い，与えられた記事見出しから
カテゴリとその予測確率を計算するプログラムを実装せよ．
"""

import numpy as np
import pandas as pd
import pickle

train = "train.txt"
valid = "valid.txt"
test = "test.txt"
train_feature = "train_feature.txt"
valid_feature = "valid_feature.txt"
test_feature = "test_feature.txt"
logistic_model = "logistic_model.sav"

#データの読み込み
train_data = pd.read_csv(train, sep="\t")
valid_data = pd.read_csv(valid, sep="\t")
test_data = pd.read_csv(test, sep="\t")
X_train = pd.read_csv(train_feature, sep="\t")
X_valid = pd.read_csv(valid_feature, sep="\t")
X_test = pd.read_csv(test_feature, sep="\t")

def score_lg(lg, X):
    #各カテゴリに対する確率を行として、それに対する最大値
    return [np.max(lg.predict_proba(X), axis=1), lg.predict(X)]

if __name__ == "__main__":
    #バイナリファイルとして保存したモデルをロードする
    with open(logistic_model, "rb") as f:
        lg = pickle.load(f)

    train_pred = score_lg(lg, X_test)
    print(train_pred)

"""
[array([0.86296065, 0.82470604, 0.9558848 , ..., 0.96284574, 0.91401294,
       0.73572053]), array(['e', 'e', 'b', ..., 'e', 'e', 't'], dtype=object)]
"""