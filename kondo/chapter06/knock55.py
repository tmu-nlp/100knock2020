"""
55. 混同行列の作成Permalink
52で学習したロジスティック回帰モデルの混同行列（confusion matrix）を，
学習データおよび評価データ上で作成せよ．
"""

import pickle
from sklearn.metrics import confusion_matrix

from knock53 import (train_data, 
                        valid_data, 
                        test_data, 
                        X_train, 
                        X_valid, 
                        X_test, 
                        score_lg, 
                        logistic_model, 
                    )

#importできること知ったので作りかけ
"""
def make_confusion_matrix(pre_array, Y_array):
    TP = 0
    TF = 0
    NP = 0
    NF = 0
    for pre, Y in zip(pre_array, Y_array):
        if pre == Y:
            c += 1
        else:
            d += 1
"""

if __name__ == "__main__":
    #バイナリファイルとして保存したモデルをロードする
    with open(logistic_model, "rb") as f:
        lg = pickle.load(f)

    train_pred = score_lg(lg, X_train)
    test_pred = score_lg(lg, X_test)
    Y_train = train_data["CATEGORY"]
    Y_test = test_data["CATEGORY"]
    train_cmatrix = confusion_matrix(Y_train.values, train_pred[1])
    test_cmatrix = confusion_matrix(Y_test.values, test_pred[1])

    print("train confusion matrix\n{}\n".format(train_cmatrix))
    print("test confusion matrix\n{}".format(test_cmatrix))

"""
train confusion matrix
[[4331  103    8   59]
 [  57 4166    2   10]
 [  95  128  491   14]
 [ 196  138    8  878]]

test confusion matrix
[[528  20   2  13]
 [ 10 518   1   1]
 [ 11  28  50   2]
 [ 37  26   1  88]]
"""