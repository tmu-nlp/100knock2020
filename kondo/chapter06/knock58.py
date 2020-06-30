"""
58. 正則化パラメータの変更Permalink
ロジスティック回帰モデルを学習するとき，正則化パラメータを調整することで，学習時の過学習（overfitting）の度合いを制御できる．
異なる正則化パラメータでロジスティック回帰モデルを学習し，学習データ，検証データ，および評価データ上の正解率を求めよ．
実験の結果は，正則化パラメータを横軸，正解率を縦軸としたグラフにまとめよ．
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt

from knock53 import (train_data, 
                        valid_data, 
                        test_data, 
                        X_train, 
                        X_valid, 
                        X_test, 
                        score_lg, 
                    )

def make_logistic_model(c):
    clf = LogisticRegression(random_state=123, max_iter=100000, penalty = "l2", C = c)
    Y_train = train_data["CATEGORY"]

    clf.fit(X_train, Y_train)
    print(clf)
    return clf

if __name__ == "__main__":
    #学習用
    """
    for i in range(-3, 4):
        lg = make_logistic_model(10**i)
        logistic_model = "logistic_model"+"{:03}".format(i)+".sav"
        #バイナリファイルにしてモデルを保存
        with open(logistic_model, "wb") as f:
            pickle.dump(lg, f)
    """

    x_train=[]
    y_train=[]
    x_valid=[]
    y_valid=[]
    x_test=[]
    y_test=[]
    for i in range(-3, 4):
        logistic_model = "logistic_model"+"{:03}".format(i)+".sav"
        with open(logistic_model, "rb") as f:
            lg = pickle.load(f)
        train_pred = score_lg(lg, X_train)
        valid_pred = score_lg(lg, X_valid)
        test_pred = score_lg(lg, X_test)
        Y_train = train_data["CATEGORY"]
        Y_valid = valid_data["CATEGORY"]
        Y_test = test_data["CATEGORY"]
        # valueをarrayとして抽出
        train_accuracy = accuracy_score(Y_train.values, train_pred[1])
        valid_accuracy = accuracy_score(Y_valid.values, valid_pred[1])
        test_accuracy = accuracy_score(Y_test.values, test_pred[1])
        x_train.append(10**i)
        y_train.append(train_accuracy)
        x_valid.append(10**i)
        y_valid.append(valid_accuracy)
        x_test.append(10**i)
        y_test.append(test_accuracy)
    
    plt.plot(x_train, y_train, label="train")
    plt.plot(x_valid, y_valid, label="valid")
    plt.plot(x_test, y_test, label="test")
    plt.ylim(0, 1.1)
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.xscale("log")
    plt.legend()    #凡例の表示
    plt.show()