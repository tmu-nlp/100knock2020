# データ処理ライブラリ「Pandas」をインポート
import pandas as pd

# 正規表現モジュール「re」をインポート
import re

def uk():
    # pandasでJSON文字列を読み込む
    data = pd.read_json('jawiki-country.json.gz', lines=True)

    # 「query()」で条件を指定
    text = data.query('title=="イギリス"')['text'].values[0]

    # 文章を改行で分割
    split = text.split('\n')

    # 抽出したい対象(メディアファイル)を定める
    rule = re.compile('File|ファイル:(.+?)\|')

    # 定めた「rule」に当てはまるものを出力する
    for file in split:
        extraction = re.findall(rule, file)

        if extraction:
            print(extraction[0])

if __name__ == '__main__':
    uk()
