"""
51. 特徴量抽出Permalink
学習データ，検証データ，評価データから特徴量を抽出し，
それぞれtrain.feature.txt，valid.feature.txt，test.feature.txtというファイル名で保存せよ．
なお，カテゴリ分類に有用そうな特徴量は各自で自由に設計せよ．
記事の見出しを単語列に変換したものが最低限のベースラインとなるであろう．
"""

"""
import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer #TF-IDF(出現頻度とレア度を掛け合わせたもの)

train = "train.txt"
valid = "valid.txt"
test = "test.txt"
train_feature = "train_feature.txt"
valid_feature = "valid_feature.txt"
test_feature = "test_feature.txt"

def get_feature(data_file_name, feature_file_name):
    with open(data_file_name, encoding = "utf-8") as data_file:
        feature = []
        categories = []
        for line in data_file:
            contents = line.split("\t")
            category = contents[0]
            title = contents[1]
            symbol = string.punctuation #記号
            table = str.maketrans(symbol, ' '*len(symbol)) #置き換える時は同じ文字数でないといけない
            title = title.translate(table) #半角記号を空白
            title = title.lower() #表記揺れ削減
            title = re.sub('[0-9]+', '0', title) #数字列を0に置換
            feature.append("{}".format(title))
            categories.append(category)
        vec_tfidf = TfidfVectorizer(min_df=10) #10回以下のは排除(データが大きくなりすぎる為)
        X = vec_tfidf.fit_transform(feature)
        df = pd.DataFrame(X.toarray(), columns=vec_tfidf.get_feature_names(), index = None)

    
    with open(feature_file_name, "w", encoding = "utf-8") as feature_file:
        df.to_csv(feature_file, sep = "\t")
    

if __name__ == "__main__":
    get_feature(train, train_feature)
    get_feature(valid, valid_feature)
    get_feature(test, test_feature)
"""

import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

train = "train.txt"
valid = "valid.txt"
test = "test.txt"
train_feature = "train_feature.txt"
valid_feature = "valid_feature.txt"
test_feature = "test_feature.txt"

#データの読み込み
train_data = pd.read_csv(train, sep="\t")
valid_data = pd.read_csv(valid, sep="\t")
test_data = pd.read_csv(test, sep="\t")

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

    return text

def preprocess():
    #縦方向に連結
    df = pd.concat([train_data, valid_data, test_data], axis=0)
    df.reset_index(drop=True)

    #前処理の実施　mapで各要素に対して前処理を実行
    df["TITLE"] = df["TITLE"].map(lambda x: preprocessing(x))
    return df

def vectorize(df):
    #関数外で定義した変数使う時に宣言しないといじったときにエラー出る
    global train_data
    global valid_data
    global test_data
    #データの再分割
    train_data = df[:len(train_data)]
    valid_data = df[len(train_data): len(train_data)+len(valid_data)]
    test_data = df[len(train_data)+len(valid_data):]

    #10回以上出現したものに限る IF-IDFを計算する単語の長さも指定
    vec_tfidf = TfidfVectorizer(min_df=10, ngram_range=(1, 2))

    #ベクトル化 学習データをベクトル化するときにトレインデータしか使わないように注意(テストにfitさせない)
    X_train = vec_tfidf.fit_transform(train_data["TITLE"])
    X_valid = vec_tfidf.transform(valid_data["TITLE"])
    X_test = vec_tfidf.transform(test_data["TITLE"])

    #ベクトルをデータフレームに変換 共通のcolumnsを利用
    X_train = pd.DataFrame(X_train.toarray(), columns=vec_tfidf.get_feature_names())
    X_valid = pd.DataFrame(X_valid.toarray(), columns=vec_tfidf.get_feature_names())
    X_test = pd.DataFrame(X_test.toarray(), columns=vec_tfidf.get_feature_names())

    X_train.to_csv(train_feature, sep = "\t", index = False)
    X_valid.to_csv(valid_feature, sep = "\t", index = False)
    X_test.to_csv(test_feature, sep = "\t", index = False)

if __name__ == "__main__":
    df = preprocess()
    vectorize(df)
