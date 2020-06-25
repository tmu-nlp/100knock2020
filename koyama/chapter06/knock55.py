# 55. 混同行列の作成
# 52で学習したロジスティック回帰モデルの混同行列（confusion matrix）を，学習データおよび評価データ上で作成せよ．

from sklearn.metrics import confusion_matrix
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

    # 混同行列を求める
    train_cm = confusion_matrix(Y_train, clf.predict(X_train))
    test_cm = confusion_matrix(Y_test, clf.predict(X_test))

    # 混同行列をデータフレームワークに変換する
    # また、カテゴリのラベルを追加する
    labels = ["b", "e", "m", "t"]
    train_cm_labeled = pd.DataFrame(train_cm, columns=labels, index=labels)
    test_cm_labeled = pd.DataFrame(test_cm, columns=labels, index=labels)

    # 混同行列を表示する
    print(f"train confusion matrix:")
    print(f"{train_cm_labeled}")
    print(f"test confusion matrix:")
    print(f"{test_cm_labeled}")

    # 処理時間 = 終了時刻 - 開始時刻
    elapsed_time = time.time() - start
    print(f"{elapsed_time} [sec]") # 58.140055894851685 [sec]

# 結果
# train confusion matrix:
#          予測したラベル
#          b     e    m    t
# 正 b  4408    57    4   33
# 解 e    12  4206    0    5
# ラ m    86   135  505    2
# ベ t   143   131    1  944
# ル
#
# test confusion matrix:
#         予測したラベル
#         b    e   m   t
# 正 b  528   23   1  11
# 解 e   11  516   0   1
# ラ m   17   17  55   2
# ベ t   32   24   2  94
# ル