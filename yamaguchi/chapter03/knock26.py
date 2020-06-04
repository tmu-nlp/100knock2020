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
    rule_1 = re.compile('\|(.+?)\s=\s*(.+)')

    # 強調マークアップの除去を定める
    rule_2 = re.compile('\'{2,}(.+?)\'{2,}')

    # 辞書オブジェクトとして初期化
    result = {}

    # 強調マークアップを除去したテキストを出力する
    for template in split:
        extraction = re.search(rule_1, template)

        if extraction:
            result[extraction[1]] = extraction[2]

        extraction = re.sub(rule_2, '\\1', template)
        print(extraction)

    print(result)

if __name__ == '__main__':
    uk()
