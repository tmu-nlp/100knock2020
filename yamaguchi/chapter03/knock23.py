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

    # 1回以上の「=」で始まり，1回以上の「=」で終わる文字列を定める．
    rule = re.compile('^=+.*=+$')

    # "="の数を数え，「=」を「」に置き換えて出力する．
    for section in split:
        if re.search(rule, section):
            extraction = section.count('=')
            print(section.replace('=',''), extraction)

if __name__ == '__main__':
    uk()
