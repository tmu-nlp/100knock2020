# 58. 正則化パラメータの変更
# ロジスティック回帰モデルを学習するとき，正則化パラメータを調整することで，学習時の過学習（overfitting）の度合いを制御できる．
# 異なる正則化パラメータでロジスティック回帰モデルを学習し，学習データ，検証データ，および評価データ上の正解率を求めよ．
# 実験の結果は，正則化パラメータを横軸，正解率を縦軸としたグラフにまとめよ．

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import time

if __name__ == "__main__":
    # 処理時間を測りたいので開始時刻を記録しておく
    start = time.time()

    # データを読み込む
    X_train = pd.read_table("train.feature.txt", header=None)
    Y_train = pd.read_table("train.txt", header=None)[1]
    X_valid = pd.read_table("valid.feature.txt", header=None)
    Y_valid = pd.read_table("valid.txt", header=None)[1]
    X_test = pd.read_table("test.feature.txt", header=None)
    Y_test = pd.read_table("test.txt", header=None)[1]

    # 正則化パラメータを変えながらモデルを学習する
    C_candidate = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
    train_acc = []
    valid_acc = []
    test_acc = []
    for c in C_candidate:
        clf = LogisticRegression(penalty="l2", solver="sag", random_state=0, C=c)
        clf.fit(X_train, Y_train)
        train_acc.append(accuracy_score(Y_train, clf.predict(X_train)))
        valid_acc.append(accuracy_score(Y_valid, clf.predict(X_valid)))
        test_acc.append(accuracy_score(Y_test, clf.predict(X_test)))

    # 正解率を表示する
    print(train_acc)
    print(valid_acc)
    print(test_acc)

    # 処理時間 = 終了時刻 - 開始時刻
    elapsed_time = time.time() - start
    print(f"{elapsed_time} [sec]") # 1050.1630408763885 [sec]

    # 正解率をグラフとして表示する
    plt.plot(C_candidate, train_acc, label="train")
    plt.plot(C_candidate, valid_acc, label="valid")
    plt.plot(C_candidate, test_acc, label="test")
    plt.xscale("log")
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

# 結果
# train:[0.760588455772114, 0.7901986506746627, 0.9429347826086957, 0.9981259370314842, 0.9992503748125937, 0.9992503748125937, 0.9992503748125937]
# valid:[0.7548725637181409, 0.7773613193403298, 0.8875562218890555, 0.9145427286356822, 0.9130434782608695, 0.9145427286356822, 0.9122938530734632]
# test :[0.7481259370314842, 0.7728635682158921, 0.8943028485757122, 0.9167916041979011, 0.9212893553223388, 0.9227886056971514, 0.9220389805097451]