"""
52. 学習Permalink
51で構築した学習データを用いて，ロジスティック回帰モデルを学習せよ．
"""

"""
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

train = "train.txt"
valid = "valid.txt"
test = "test.txt"
train_feature = "train_feature.txt"
valid_feature = "valid_feature.txt"
test_feature = "test_feature.txt"

#trainデータで作成
def make_logistic_model():
    feature = []

    with open(train_feature, encoding="utf-8") as train_feature_data:
        x_train = pd.read_table(train_feature_data)

    with open(train, encoding="utf-8") as train_data:
        y_train = pd.read_table(train_data, header = None)[0]

    clf = LogisticRegression(max_iter=10000)
    clf.fit(x_train, y_train)
    return clf
    

if __name__ == "__main__":
    lg = make_logistic_model()
    lg.predict_proba()
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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

def make_logistic_model():
    clf = LogisticRegression(random_state=123, max_iter=100000)
    Y_train = train_data["CATEGORY"]

    clf.fit(X_train, Y_train)
    print(clf)
    return clf

if __name__ == "__main__":
    lg = make_logistic_model()
    #バイナリファイルにしてモデルを保存
    with open(logistic_model, "wb") as f:
        pickle.dump(lg, f)