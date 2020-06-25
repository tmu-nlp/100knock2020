# 59. ハイパーパラメータの探索
# 学習アルゴリズムや学習パラメータを変えながら，カテゴリ分類モデルを学習せよ．
# 検証データ上の正解率が最も高くなる学習アルゴリズム・パラメータを求めよ．
# また，その学習アルゴリズム・パラメータを用いたときの評価データ上の正解率を求めよ．

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from tqdm import tqdm
import pandas as pd
import time

def output_info(model, valid_acc, test_acc):
    with open("result59.txt", "a") as ouput_file:
        print(f"algorithm     : {model}", file=ouput_file)
        print(f"valid accuracy: {valid_acc}", file=ouput_file)
        print(f"test accuracy : {test_acc}", file=ouput_file)

if __name__ == "__main__":
    # データを読み込む
    X_train = pd.read_table("train.feature.txt", header=None)
    Y_train = pd.read_table("train.txt", header=None)[1]
    X_valid = pd.read_table("valid.feature.txt", header=None)
    Y_valid = pd.read_table("valid.txt", header=None)[1]
    X_test = pd.read_table("test.feature.txt", header=None)
    Y_test = pd.read_table("test.txt", header=None)[1]

    # 標準化
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_valid = sc.transform(X_valid)
    X_test = sc.transform(X_test)

    # # 主成分分析
    # # 次元圧縮前の特徴量は13109次元
    # pca = PCA(n_components=512)
    # X_train = pca.fit_transform(X_train)
    # X_valid = pca.transform(X_valid)
    # X_test = pca.transform(X_test)

    # 必要な情報を格納する変数
    best_model = None
    best_acc = -1 # valid上の最大の正解率

    # ロジスティック回帰
    C_candidate = [1e-2, 1e-1, 1e0, 1e1, 1e2]
    for c in tqdm(C_candidate):
        clf = LogisticRegression(penalty="l1", solver="saga", random_state=0, C=c)
        clf.fit(X_train, Y_train)
        valid_acc = accuracy_score(Y_valid, clf.predict(X_valid))
        if best_acc < valid_acc:
            best_model = clf
            best_acc = valid_acc
    test_acc = accuracy_score(Y_test, best_model.predict(X_test)) # test上で評価する
    output_info(best_model, best_acc, test_acc)                   # 結果を出力する
    best_acc = -1                                                 # validの正解率をリセットする

    # SVM
    C_candidate = [1e-2, 1e-1, 1e0, 1e1, 1e2]
    for c in tqdm(C_candidate):
        clf = SVC(kernel="linear", C=c, class_weight="balanced", random_state=0)
        clf.fit(X_train, Y_train)
        valid_acc = accuracy_score(Y_valid, clf.predict(X_valid))
        if best_acc < valid_acc:
            best_model = clf
            best_acc = valid_acc
    test_acc = accuracy_score(Y_test, best_model.predict(X_test))
    output_info(best_model, best_acc, test_acc)
    best_acc = -1

    # # カーネルSVM
    # # 非常に時間がかかる
    # C_candidate = [1e-2, 1e-1, 1e0, 1e1, 1e2]
    # gamma_candidate = [1e-2, 1e-1, 1e0, 1e1, 1e2]
    # for c in tqdm(C_candidate):
    #     for gam in tqdm(gamma_candidate):
    #         clf = SVC(kernel="rbf", C=c, gamma=gam, class_weight="balanced", random_state=0)
    #         clf.fit(X_train, Y_train)
    #         valid_acc = accuracy_score(Y_valid, clf.predict(X_valid))
    #         if best_acc < valid_acc:
    #             best_model = clf
    #             best_acc = valid_acc
    # test_acc = accuracy_score(Y_test, best_model.predict(X_test))
    # output_info(best_model, best_acc, test_acc)
    # best_acc = -1

    # ランダムフォレスト
    max_depth_candidate = [16, 32, 64, 128, 256]
    for m in max_depth_candidate:
        clf = RandomForestClassifier(max_depth=m, random_state=0)
        clf.fit(X_train, Y_train)
        valid_acc = accuracy_score(Y_valid, clf.predict(X_valid))
        if best_acc < valid_acc:
            best_model = clf
            best_acc = valid_acc
    test_acc = accuracy_score(Y_test, best_model.predict(X_test))
    output_info(best_model, best_acc, test_acc)
    best_acc = -1

# 結果
# algorithm     : LogisticRegression(C=0.1, penalty='l1', random_state=0, solver='saga')
# valid accuracy: 0.9250374812593704
# test accuracy : 0.9250374812593704
