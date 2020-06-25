"""
50. データの入手・整形Permalink
News Aggregator Data Setをダウンロードし、
以下の要領で学習データ（train.txt），検証データ（valid.txt），評価データ（test.txt）を作成せよ．

1. ダウンロードしたzipファイルを解凍し，readme.txtの説明を読む．
2. 情報源（publisher）が”Reuters”, “Huffington Post”, “Businessweek”,
“Contactmusic.com”, “Daily Mail”の事例（記事）のみを抽出する．
3. 抽出された事例をランダムに並び替える．
4. 抽出された事例の80%を学習データ，残りの10%ずつを検証データと評価データに分割し，
それぞれtrain.txt，valid.txt，test.txtというファイル名で保存する．
ファイルには，１行に１事例を書き出すこととし，カテゴリ名と記事見出しのタブ区切り形式とせよ（このファイルは後に問題70で再利用する）．
学習データと評価データを作成したら，各カテゴリの事例数を確認せよ．
"""

"""
import random
import pandas as pd

data = "./NewsAggregatorDataset/newsCorpora.csv"
train = "train.txt"
valid = "valid.txt"
test = "test.txt"

def get_data(file_name)->list:
    articles = []
    with open(file_name, encoding = "utf-8") as data_file:
        for line in data_file:
            contents = line.split("\t")
            id_num = contents[0]
            title = contents[1]
            url = contents[2]
            publisher = contents[3]
            category = contents[4]
            story = contents[5]
            hostname = contents[6]
            timestamp = contents[7]
            if (publisher == "Reuters"
                or publisher == "Huffington Post"
                or publisher == "Businessweek"
                or publisher == "Contactmusic.com"
                or publisher == "Daily Mail"
            ):
                articles.append(category+"\t"+title)
    return articles

def split_data(data_list):
    l = len(data_list)
    train_len = int(l*0.8)
    valid_len = test_len = int(l*0.1)
    if l - (train_len+2*valid_len) == 1:
        train_len += 1
    elif l - (train_len+2*valid_len) == 2:
        valid_len += 1
        test_len += 1
    train_data = data_list[: train_len]
    valid_data = data_list[train_len: train_len+valid_len]
    test_data = data_list[train_len+valid_len:]
    print("train: {}\nvalid: {}\ntest: {}".format(len(train_data), len(valid_data), len(test_data)))
    train_data = "\n".join(train_data)
    valid_data = "\n".join(valid_data)
    test_data = "\n".join(test_data)
    with open(train, "w", encoding="utf-8") as train_file,\
            open(valid, "w", encoding="utf-8") as valid_file,\
            open(test, "w", encoding="utf-8") as test_file:
        train_file.writelines(train_data)
        valid_file.write(valid_data)
        test_file.write(test_data)

if __name__ == "__main__":
    random.seed(0)
    shuffled_data = random.sample(get_data(data), len(get_data(data)))
    split_data(shuffled_data)
"""

import pandas as pd
from sklearn.model_selection import train_test_split

data = "./NewsAggregatorDataset/newsCorpora_re.csv"
train = "train.txt"
valid = "valid.txt"
test = "test.txt"

def get_data():
    #データの読み込み tab区切りで左からフォーマット名
    df = pd.read_csv(data, header = None, sep="\t", names=["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"])
    #データの抽出
    #isin()
    #loc[x, y] データの位置(x行目y列)を指定
    df = df.loc[df["PUBLISHER"].isin(["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]), ["TITLE", "CATEGORY"]]

    #データの分轄(シードは固定、CATEGORYが均等になるように)
    train_data, valid_test_data = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df["CATEGORY"])
    print(len(df))
    valid_data, test_data = train_test_split(valid_test_data, test_size=0.5, shuffle=True, random_state=123, stratify=valid_test_data["CATEGORY"])

    #データの保存
    train_data.to_csv(train, sep="\t", index=False)
    valid_data.to_csv(valid, sep="\t", index=False)
    test_data.to_csv(test, sep="\t", index=False)

    print('[学習データ]')
    print(train_data["CATEGORY"].value_counts())
    print('valid_data[検証データ]')
    print(valid_data["CATEGORY"].value_counts())
    print('test_data[テストデータ]')
    print(test_data["CATEGORY"].value_counts())

if __name__ == "__main__":
    get_data()