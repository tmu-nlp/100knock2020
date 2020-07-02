"""
54. 正解率の計測Permalink
52で学習したロジスティック回帰モデルの正解率を，学習データおよび評価データ上で計測せよ．
"""

import pickle

from knock53 import (train_data, 
                        valid_data, 
                        test_data, 
                        X_train, 
                        X_valid, 
                        X_test, 
                        score_lg, 
                        logistic_model, 
                    )

def model_accuracy(pre_array, Y_array):
    c = 0
    d = 0
    for pre, Y in zip(pre_array, Y_array):
        if pre == Y:
            c += 1
        else:
            d += 1
    return float(c)/float(c+d)

if __name__ == "__main__":
    #バイナリファイルとして保存したモデルをロードする
    with open(logistic_model, "rb") as f:
        lg = pickle.load(f)

    train_pred = score_lg(lg, X_train)
    test_pred = score_lg(lg, X_test)
    Y_train = train_data["CATEGORY"]
    Y_test = test_data["CATEGORY"]
    # valueをarrayとして抽出
    train_accuracy = model_accuracy(train_pred[1], Y_train.values)
    test_accuracy = model_accuracy(test_pred[1], Y_test.values)
    print("train accuracy: {}".format(train_accuracy))
    print("test accuracy: {}".format(test_accuracy))

#from sklearn.metrics import accuracy_score  があった...
#accuracy_score(y_true, y_pred)で使う感じ、自分が作ったのと逆...

"""
train accuracy: 0.9234369150131037
test accuracy: 0.8862275449101796
"""