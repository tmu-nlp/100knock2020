"""
71. 単層ニューラルネットワークによる予測Permalink
問題70で保存した行列を読み込み，学習データについて以下の計算を実行せよ．

ŷ 1=softmax(x1W),Ŷ =softmax(X[1:4]W)
ただし，softmaxはソフトマックス関数，X[1:4]∈ℝ4×dは特徴ベクトルx1,x2,x3,x4を縦に並べた行列である．

X[1:4]=⎛⎝⎜⎜⎜⎜x1x2x3x4⎞⎠⎟⎟⎟⎟
行列W∈ℝd×Lは単層ニューラルネットワークの重み行列で，ここではランダムな値で初期化すればよい（問題73以降で学習して求める）．
なお，ŷ 1∈ℝLは未学習の行列Wで事例x1を分類したときに，各カテゴリに属する確率を表すベクトルである． 
同様に，Ŷ ∈ℝn×Lは，学習データの事例x1,x2,x3,x4について，各カテゴリに属する確率を行列として表現している
"""

import numpy as np

X_train_data = "X_train.npy"
Y_train_data = "Y_train.npy"
X_valid_data = "X_valid.npy"
Y_valid_data = "Y_valid.npy"
X_test_data = "X_test.npy"
Y_test_data = "Y_test.npy"

d = 300
L = 4

def softmax(x):
    u = np.sum(np.exp(x))
    return np.exp(x)/u

def probably(X, w):
    p = []
    for x in X:
        p.append(softmax(np.dot(x, w)))
    return np.array(p)

if __name__ == "__main__":
    np.random.seed(7)
    W = np.random.rand(d, L)
    X = np.load(file=X_train_data)
    x1 = X[0:1]
    X_1_4 = X[:4]
    y_hat = probably(x1, W)
    Y_hat = probably(X_1_4, W)

    print("y_hat")
    print(y_hat)
    print("Y_hat")
    print(Y_hat)

"""
y_hat
[[0.14312437 0.35936813 0.27431715 0.22319036]]
Y_hat
[[0.14312437 0.35936813 0.27431715 0.22319036]
 [0.20668898 0.32553434 0.26566188 0.2021148 ]
 [0.2743068  0.20786322 0.32814228 0.18968769]
 [0.22264475 0.23470634 0.29803636 0.24461255]]
"""