# 51. 特徴量抽出
# 学習データ，検証データ，評価データから特徴量を抽出し，それぞれtrain.feature.txt，valid.feature.txt，test.feature.txtというファイル名で保存せよ．
# なお，カテゴリ分類に有用そうな特徴量は各自で自由に設計せよ．
# 記事の見出しを単語列に変換したものが最低限のベースラインとなるであろう．

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from stemming.porter2 import stem
import pandas as pd
import numpy as np
import joblib
import time
import re

# 前処理を行う関数
def preprocessor(text):
    # 文字を全て小文字にする
    text = text.lower()

    # 数字を全て0にする
    text = re.sub(r"\d+", "0", text)

    # トークナイズ
    # word_tokenize()を適用するとリストが返ってくるので、" ".join()で結合する
    text = " ".join(word_tokenize(text))

    # ステミング
    text = stem(text)

    return text

if __name__ == "__main__":
    # 処理時間を測りたいので開始時刻を記録しておく
    start = time.time()

    # データを読み込む
    XY_train = pd.read_csv("train.txt", header=None, sep="\t")
    XY_valid = pd.read_csv("valid.txt", header=None, sep="\t")
    XY_test = pd.read_csv("test.txt", header=None, sep="\t")

    # 列の名前を設定する
    columns_name = ["TITLE", "CATEGORY"]
    XY_train.columns = columns_name
    XY_valid.columns = columns_name
    XY_test.columns = columns_name

    # TITLEの文字列に対して、前処理を行う
    XY_train["TITLE"] = XY_train["TITLE"].apply(preprocessor)
    XY_valid["TITLE"] = XY_valid["TITLE"].apply(preprocessor)
    XY_test["TITLE"] = XY_test["TITLE"].apply(preprocessor)

    # TF-IDFを計算する
    vectorizer = TfidfVectorizer(use_idf=True, norm="l2", smooth_idf=True)
    vectorizer.fit(XY_train["TITLE"]) # trainのみを用いて訓練する
    train_tfidf = vectorizer.transform(XY_train["TITLE"]) # 訓練したものを適用する
    valid_tfidf = vectorizer.transform(XY_valid["TITLE"]) # 訓練したものを適用する
    test_tfidf = vectorizer.transform(XY_test["TITLE"])   # 訓練したものを適用する

    # データを保存する
    pd.DataFrame(data=train_tfidf.toarray()).to_csv("train.feature.txt", sep="\t", index=False, header=None)
    pd.DataFrame(data=valid_tfidf.toarray()).to_csv("valid.feature.txt", sep="\t", index=False, header=None)
    pd.DataFrame(data=test_tfidf.toarray()).to_csv("test.feature.txt", sep="\t", index=False, header=None)
    joblib.dump(vectorizer.vocabulary_, "vocabulary_.joblib") # knock57で使用する


    # 処理時間 = 終了時刻 - 開始時刻
    elapsed_time = time.time() - start
    print(f"{elapsed_time} [sec]") # 149.65297412872314 [sec]
