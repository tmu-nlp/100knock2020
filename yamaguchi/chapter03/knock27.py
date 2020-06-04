# データ処理ライブラリ「Pandas」をインポート
import pandas as pd

# 正規表現モジュール「re」をインポート
import re

def uk():
    # pandasでJSON文字列を読み込む
    data = pd.read_json('jawiki-country.json.gz', lines=True)

    # 「query()」で条件を指定
    text = data.query('title=="イギリス"')['text'].values[0]

    # 抽出したい対象(「基礎情報」のテンプレート)を定める
    rule_1 = re.compile('\|(.+?)\s=\s*(.+)')

    # 強調マークアップの除去を定める
    rule_2 = re.compile('\'{2,}(.+?)\'{2,}')

    # 内部リンクの除去を定める
    rule_3 = re.compile('\[\[(.+?)\]\]')

    # 定めたものに従って除去する
    extraction = re.sub(rule_2, '\\1', text)
    extraction = re.sub(rule_3, '\\1', extraction)

    # 文章を改行で分割
    split = extraction.split('\n')

    # 辞書オブジェクトとして初期化
    result = {}

    # 強調マークアップを除去したテキストを出力する
    for template in split:
        extraction = re.search(rule_1, template)

        if extraction:
            result[extraction[1]] = extraction[2]

    print(result)

if __name__ == '__main__':
    uk()
