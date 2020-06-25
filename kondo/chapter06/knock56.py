"""
56. 適合率，再現率，F1スコアの計測Permalink
52で学習したロジスティック回帰モデルの適合率，再現率，F1スコアを，評価データ上で計測せよ．
カテゴリごとに適合率，再現率，F1スコアを求め，
カテゴリごとの性能をマイクロ平均（micro-average）とマクロ平均（macro-average）で統合せよ．
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (precision_score, 
                                recall_score, 
                                f1_score
                            )

from knock53 import (train_data, 
                        valid_data, 
                        test_data, 
                        X_train, 
                        X_valid, 
                        X_test, 
                        score_lg, 
                        logistic_model, 
                    )

if __name__ == "__main__":
    #バイナリファイルとして保存したモデルをロードする
    with open(logistic_model, "rb") as f:
        lg = pickle.load(f)

    test_pred = score_lg(lg, X_test)
    Y_test = test_data["CATEGORY"]
    #マルチクラスラベルなのでaverageの設定が必須 Noneとすると各ラベルのスコアがarrayで返される
    test_precision = precision_score(Y_test.values, test_pred[1], average=None)
    test_precision = np.append(test_precision, precision_score(Y_test.values, test_pred[1], average="micro"))
    test_precision = np.append(test_precision, precision_score(Y_test.values, test_pred[1], average="macro"))

    test_recall = recall_score(Y_test.values, test_pred[1], average=None, labels=["b", "t", "e", "m"])
    test_recall = np.append(test_recall, recall_score(Y_test.values, test_pred[1], average="micro"))
    test_recall = np.append(test_recall, recall_score(Y_test.values, test_pred[1], average="macro"))

    test_f1 = f1_score(Y_test.values, test_pred[1], average=None, labels=["b", "t", "e", "m"])
    test_f1 = np.append(test_f1, f1_score(Y_test.values, test_pred[1], average="micro"))
    test_f1 = np.append(test_f1, f1_score(Y_test.values, test_pred[1], average="macro"))

    #indexは指定したラベルに合わせる(ラベル指定しないとアルファベット順な気がする)
    scores = pd.DataFrame({"precision": test_precision,
                            "recall": test_recall,
                            "f1": test_f1},
                            index=["b", "t", "e", "m", "micro", "macro"])
    print(scores)

"""
       precision    recall        f1
b       0.901024  0.937833  0.919060
t       0.875000  0.578947  0.687500
e       0.925926  0.977358  0.923351
m       0.846154  0.549451  0.689655
micro   0.886228  0.886228  0.886228
macro   0.887026  0.760897  0.804892
"""