"""
70. 単語ベクトルの和による特徴量Permalink
問題50で構築した学習データ，検証データ，評価データを行列・ベクトルに変換したい．
例えば，学習データについて，すべての事例xiの特徴ベクトルxiを並べた行列Xと，正解ラベルを並べた行列（ベクトル）Yを作成したい．

X=⎛⎝⎜⎜⎜⎜x1x2…xn⎞⎠⎟⎟⎟⎟∈ℝn×d,Y=⎛⎝⎜⎜⎜⎜y1y2…yn⎞⎠⎟⎟⎟⎟∈ℕn
ここで，nは学習データの事例数であり，xi∈ℝdとyi∈ℕはそれぞれ，i∈{1,…,n}番目の事例の特徴量ベクトルと正解ラベルを表す． 
なお，今回は「ビジネス」「科学技術」「エンターテイメント」「健康」の4カテゴリ分類である．
ℕ<4で4未満の自然数（0を含む）を表すことにすれば，任意の事例の正解ラベルyiはyi∈ℕ<4で表現できる． 
以降では，ラベルの種類数をLで表す（今回の分類タスクではL=4である）．

i番目の事例の特徴ベクトルxiは，次式で求める．

xi=1Ti∑t=1Tiemb(wi,t)
ここで，i番目の事例はTi個の（記事見出しの）単語列(wi,1,wi,2,…,wi,Ti)から構成され，
emb(w)∈ℝdは単語wに対応する単語ベクトル（次元数はd）である．
すなわち，i番目の事例の記事見出しを，その見出しに含まれる単語のベクトルの平均で表現したものがxiである．
今回は単語ベクトルとして，問題60でダウンロードしたものを用いればよい．
300次元の単語ベクトルを用いたので，d=300である．

i番目の事例のラベルyiは，次のように定義する．

yi=⎧⎩⎨⎪⎪0123(記事xiが「ビジネス」カテゴリの場合)(記事xiが「科学技術」カテゴリの場合)(記事xiが「エンターテイメント」カテゴリの場合)(記事xiが「健康」カテゴリの場合)
なお，カテゴリ名とラベルの番号が一対一で対応付いていれば，上式の通りの対応付けでなくてもよい．

以上の仕様に基づき，以下の行列・ベクトルを作成し，ファイルに保存せよ．

学習データの特徴量行列: Xtrain∈ℝNt×d
学習データのラベルベクトル: Ytrain∈ℕNt
検証データの特徴量行列: Xvalid∈ℝNv×d
検証データのラベルベクトル: Yvalid∈ℕNv
評価データの特徴量行列: Xtest∈ℝNe×d
評価データのラベルベクトル: Ytest∈ℕNe
なお，Nt,Nv,Neはそれぞれ，学習データの事例数，検証データの事例数，評価データの事例数である．
"""


import csv
import pickle
import re
import string
import numpy as np
from nltk.tokenize import word_tokenize

train_file = "../chapter06/train.csv"
valid_file = "../chapter06/valid.csv"
test_file = "../chapter06/test.csv"

X_train_file = "X_train"
Y_train_file = "Y_train"
X_valid_file = "X_valid"
Y_valid_file = "Y_valid"
X_test_file = "X_test"
Y_test_file = "Y_test"

model_file = "../chapter07/model.sav"
with open(model_file, "rb") as file_model:
    model = pickle.load(file_model)

def preprocessing(text):
    #記号
    symbol = string.punctuation

    #記号をスペースに変換　maketransの置き換えは同じ文字数じゃないと不可
    table = str.maketrans(symbol, ' '*len(symbol))
    text = text.translate(table)

    #小文字に統一
    text = text.lower()
    #数字列を0に変換
    text = re.sub("[0-9]+", "0", text)

    text = word_tokenize(text)

    return text

def to_vec(file_name, X_file, Y_file):
    with open(file_name) as f:
        reader = csv.reader(f, delimiter="\t")
        l = [row for row in reader]
        l = l[1:]
        category = ["b", "t", "e", "m"]
        x = []
        y = []
        for i, row in enumerate(l):
            y.append(category.index(row[1]))
            words = preprocessing(row[0])
            sum = 0
            t = 0
            for word in words:
                try:            #モデルになかったらスルー
                    sum += model[word]
                    t += 1
                except KeyError:
                    continue
            x.append(sum/t)

        print(len(x))

        np.set_printoptions(precision=10, suppress=True, linewidth=5000)

        np.save(X_file, x)
        np.save(Y_file, y)

if __name__ == "__main__":
    to_vec(train_file, X_train_file, Y_train_file)
    to_vec(valid_file, X_valid_file, Y_valid_file)
    to_vec(test_file, X_test_file, Y_test_file)