# 54. 正解率の計測
# 52で学習したロジスティック回帰モデルの正解率を，学習データおよび評価データ上で計測せよ．

from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import time

if __name__ == "__main__":
    # 処理時間を測りたいので開始時刻を記録しておく
    start = time.time()

    # データを読み込む
    X_train = pd.read_table("train.feature.txt", header=None)
    Y_train = pd.read_table("train.txt", header=None)[1]
    X_test = pd.read_table("test.feature.txt", header=None)
    Y_test = pd.read_table("test.txt", header=None)[1]

    # モデルを読み込む
    clf = joblib.load("model.joblib")

    # 正解率を求める
    train_acc = accuracy_score(Y_train, clf.predict(X_train))
    test_acc = accuracy_score(Y_test, clf.predict(X_test))

    # 正解率を表示する
    print(f"train accuracy: {train_acc}")  # train accuracy: 0.9429347826086957
    print(f"test  accuracy: {test_acc}")   # test  accuracy: 0.8943028485757122

    # 処理時間 = 終了時刻 - 開始時刻
    elapsed_time = time.time() - start
    print(f"{elapsed_time} [sec]") # 52.70988988876343 [sec]
