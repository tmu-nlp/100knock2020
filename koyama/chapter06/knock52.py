# 52. 学習
# 51で構築した学習データを用いて，ロジスティック回帰モデルを学習せよ．

from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib
import time

if __name__ == "__main__":
    # 処理時間を測りたいので開始時刻を記録しておく
    start = time.time()

    # データを読み込む
    X_train = pd.read_table("train.feature.txt", header=None)
    Y_train = pd.read_table("train.txt", header=None)[1] # 正解ラベルのみを取り出す

    # モデルを学習する
    # penalty: 正則化手法
    # solver: 最適解の探索手法
    clf = LogisticRegression(penalty="l2", solver="sag", random_state=0)
    clf.fit(X_train, Y_train)

    # モデルを保存する
    joblib.dump(clf, "model.joblib")

    # 処理時間 = 終了時刻 - 開始時刻
    elapsed_time = time.time() - start
    print(f"{elapsed_time} [sec]") # 146.86059093475342 [sec]

