"""
50. データの入手・整形
News Aggregator Data Setをダウンロードし、以下の要領で
学習データ（train.txt），検証データ（valid.txt），評価データ（test.txt）を作成せよ．

1. ダウンロードしたzipファイルを解凍し，readme.txtの説明を読む．
2. 情報源（publisher）が”Reuters”, “Huffington Post”, “Businessweek”,
   “Contactmusic.com”, “Daily Mail”の事例（記事）のみを抽出する．
3. 抽出された事例をランダムに並び替える．
4. 抽出された事例の80%を学習データ，残りの10%ずつを検証データと評価データに分割し，
   それぞれtrain.txt，valid.txt，test.txtというファイル名で保存する．
   ファイルには，１行に１事例を書き出すこととし，カテゴリ名と記事見出しのタブ区切り形式とせよ
   （このファイルは後に問題70で再利用する）．

学習データと評価データを作成したら，各カテゴリの事例数を確認せよ．

[MEMO]
2015 年版の knock70 に対応
"""
import os
import random
import sys
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer  # noqa: E402 isort:skip

random.seed(123)

filepath = "./NewsAggregatorDataset/newsCorpora.csv"
field_names = (
    "ID",
    "TITLE",
    "URL",
    "PUBLISHER",
    "CATEGORY",
    "STORY",
    "HOSTNAME",
    "TIMESTAMP",
)
wants = [
    "Reuters",
    "Huffington Post",
    "Businessweek",
    "Contactmusic.com",
    "Daily Mail",
]
categories = {
    "b": "business",
    "t": "science and technology",
    "e": "entertainment",
    "m": "health",
}

data = []
for line in open(filepath):
    cols = dict(zip(field_names, line.split("\t")))
    if cols["PUBLISHER"] in wants:
        data.append(cols)
random.shuffle(data)

total = len(data)
unit = total // 10
datasets = {
    "train": range(0, total - unit * 2),
    "valid": range(total - unit * 2, total - unit),
    "test": range(total - unit, total),
}


def write_file(filename, rng):
    category_cnter = defaultdict(int)
    with open(filename + ".txt", "w") as f:
        for i in rng:
            print(data[i]["CATEGORY"], data[i]["TITLE"], sep="\t", file=f)
            category_cnter[data[i]["CATEGORY"]] += 1
    with Renderer(filename) as out:
        for tag, name in categories.items():
            out.result(name, category_cnter[tag])


for dataset in datasets.items():
    write_file(*dataset)
