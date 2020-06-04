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

    # 抽出したい対象(「基礎情報」のテンプレート)を定める
    rule = re.compile('\|(.+?)\s=\s*(.+)')

    # 辞書オブジェクトとして初期化
    result = {}

    # 定めた「rule」に当てはまるものを出力する
    for template in split:
        extraction = re.search(rule, template)

        if extraction:
            result[extraction[1]] = extraction[2]

    print(result)

if __name__ == '__main__':
    uk()
