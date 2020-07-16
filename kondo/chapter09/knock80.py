"""
80. ID番号への変換Permalink
問題51で構築した学習データ中の単語にユニークなID番号を付与したい．
学習データ中で最も頻出する単語に1，2番目に頻出する単語に2，……といった方法で，学習データ中で2回以上出現する単語にID番号を付与せよ．
そして，与えられた単語列に対して，ID番号の列を返す関数を実装せよ．
ただし，出現頻度が2回未満の単語のID番号はすべて0とせよ．
"""

import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

train = "../chapter06/train.txt"
valid = "../chapter06/valid.txt"
test = "../chapter06/test.txt"
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

    return text

def preprocess():
    #縦方向に連結
    df = pd.concat([train_data, valid_data, test_data], axis=0)
    df.reset_index(drop=True)

    #前処理の実施　mapで各要素に対して前処理を実行
    df["TITLE"] = df["TITLE"].map(lambda x: preprocessing(x))
    return df

def dist(df):
    #関数外で定義した変数使う時に宣言しないといじったときにエラー出る
    global train_data
    global valid_data
    global test_data
    #データの再分割
    train_data = df[:len(train_data)]
    valid_data = df[len(train_data): len(train_data)+len(valid_data)]
    test_data = df[len(train_data)+len(valid_data):]
    return train_data, valid_data, test_data

def make_id(data):
    word_cnt = defaultdict(int)
    for text in data["TITLE"]:
        for word in text.split():
            word_cnt[word] += 1
    word_cnt = sorted(word_cnt.items(), key=lambda x:x[1], reverse=True)

    ids = defaultdict(int)
    for i, (word, cnt) in enumerate(word_cnt, start=1):
        if cnt > 1:
            ids[word] = i
    return ids

def translate_to_id(text):
    ids = {}
    V = 1   #次以降のためにidの数
    with open("id_file", encoding="utf-8") as f:
        for line in f:
            id, word = line.split()
            ids[word] = int(id)
            if int(id) > 0: V += 1
    text = preprocessing(text)
    trans_text = []
    for word in text.split():
        if word in ids:
            trans_text.append(ids[word])
        else: trans_text.append(0)
    return " ".join(map(str, trans_text)), V

if __name__ == "__main__":
    df = preprocess()
    tre, val, tes = dist(df)
    ids = make_id(tre)
    with open("id_file", "w", encoding="utf-8") as f:
        for value, id in ids.items():
            f.write(f"{id:04}\t{value}\n")

    id_text, v = translate_to_id("I hope at least I taste good.")
    print(f"id_text: {id_text}\nid登録単語数{v}")

"""
id_text: 106 2050 15 1853 106 7247 366
id登録単語数7646
"""