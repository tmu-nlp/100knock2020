# 70. 単語ベクトルの和による特徴量
# 問題50で構築した学習データ，検証データ，評価データを行列・ベクトルに変換したい．
# 例えば，学習データについて，すべての事例xiの特徴ベクトルxiを並べた行列Xと，正解ラベルを並べた行列（ベクトル）Yを作成したい．

from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import joblib

def culcSwem(row):
    global model

    # 単語ベクトルに変換する
    swem = []
    for w in row["TITLE"].split():
        if w in model.vocab:
            swem.append(model[w])
        else:
            swem.append(np.zeros(shape=(model.vector_size,)))

    # 平均に変換する
    swem = np.mean(np.array(swem), axis=0)

    return swem

if __name__ == "__main__":
    # データを読み込む
    X_train = pd.read_table("train.txt", header=None)
    X_valid = pd.read_table("valid.txt", header=None)
    X_test = pd.read_table("test.txt", header=None)
    model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

    # カラム名を設定する
    use_cols = ["TITLE", "CATEGORY"]
    X_train.columns = use_cols
    X_valid.columns = use_cols
    X_test.columns = use_cols

    # train、valid、testをまとめる
    data = pd.concat([X_train, X_valid, X_test]).reset_index(drop=True)

    # 特徴ベクトルを作成する
    swemVec = data.apply(culcSwem, axis=1)

    # それぞれのカテゴリを自然数に変換する
    y_data = data["CATEGORY"].map({"b": 0, "e": 1, "t": 2, "m": 3})

    # それぞれの事例数を得る
    n_train = len(X_train)
    n_valid = len(X_valid)
    n_test = len(X_test)

    # train、valid、testに分割する
    X_train = np.array(list(swemVec.values)[:n_train])
    X_valid = np.array(list(swemVec.values)[n_train:n_train + n_valid])
    X_test = np.array(list(swemVec.values)[n_train + n_valid:])
    y_train = y_data.values[:n_train]
    y_valid = y_data.values[n_train:n_train + n_valid]
    y_test = y_data.values[n_train + n_valid:]

    # データを保存する
    joblib.dump(X_train, "X_train.joblib")
    joblib.dump(X_valid, "X_valid.joblib")
    joblib.dump(X_test, "X_test.joblib")
    joblib.dump(y_train, "y_train.joblib")
    joblib.dump(y_valid, "y_valid.joblib")
    joblib.dump(y_test, "y_test.joblib")