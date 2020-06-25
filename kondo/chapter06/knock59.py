"""
59. ハイパーパラメータの探索Permalink
学習アルゴリズムや学習パラメータを変えながら，カテゴリ分類モデルを学習せよ．
検証データ上の正解率が最も高くなる学習アルゴリズム・パラメータを求めよ．
また，その学習アルゴリズム・パラメータを用いたときの評価データ上の正解率を求めよ．
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

from knock53 import (train_data, 
                        valid_data, 
                        test_data, 
                        X_train, 
                        X_valid, 
                        X_test, 
                        score_lg, 
                    )

def select_best_model():
    Y_train = train_data["CATEGORY"]
    Y_valid = valid_data["CATEGORY"]
    max_accuracy = 0
    for solver in ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]:
        clf = LogisticRegression(random_state=123, max_iter=100000, solver=solver)
        clf.fit(X_train, Y_train)
        valid_pred = score_lg(clf, X_valid)
        valid_accuracy = accuracy_score(Y_valid.values, valid_pred[1])
        if valid_accuracy > max_accuracy:
            max_accuracy = valid_accuracy
            best_solver = solver
        print("solver: {} valid_accuracy: {}".format(solver, valid_accuracy))
    print("best_solver: {}".format(best_solver))

    return clf

if __name__ == "__main__":
        lg = select_best_model()
        test_pred = score_lg(lg, X_test)
        Y_test = test_data["CATEGORY"]
        test_accuracy = accuracy_score(Y_test.values, test_pred[1])
        print("test_accuracy: {}".format(test_accuracy))

"""
solver: newton-cg valid_accuracy: 0.8764970059880239
solver: lbfgs valid_accuracy: 0.8764970059880239
solver: liblinear valid_accuracy: 0.8720059880239521
solver: sag valid_accuracy: 0.8764970059880239
solver: saga valid_accuracy: 0.8764970059880239
best_solver: newton-cg
test_accuracy: 0.8862275449101796
"""