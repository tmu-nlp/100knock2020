# データ処理ライブラリ「Pandas」をインポート
import pandas as pd

# 正規表現モジュール「re」をインポート
import re

# HTTP通信ライブラリ(Webサイトの画像の収集のため)をインポート
import requests

def uk():
    # pandasでJSON文字列を読み込む
    data_1 = pd.read_json('jawiki-country.json.gz', lines=True)

    # 「query()」で条件を指定
    text = data_1.query('title=="イギリス"')['text'].values[0]

    # 文章を改行で分割
    split = text.split('\n')

    # 辞書オブジェクトとして初期化
    result = {}

    # 抽出したい対象(「基礎情報」のテンプレート)を定める
    rule = re.compile('\|(.+?)\s=\s*(.+)')

    # 「rule」に従ってテンプレートを作成
    for template in split:
        extraction = re.search(rule, template)

        if extraction:
            result[extraction[1]] = extraction[2]

    # 以下，国旗画像のURLを取得する．
    session = requests.Session()

    URL = "https://commons.wikimedia.org/w/api.php"

    PARAMS = {
        "action": "query",
        "format": "json",
        "titles": "File:" + result['国旗画像'],
        "prop": "imageinfo",
        "iiprop":"url"
    }

    progress = session.get(url = URL, params = PARAMS)

    data_2 = progress.json()

    PAGES = data_2['query']['pages']

    for i, j in PAGES.items():
        print(j['imageinfo'][0]['url'])

if __name__ == '__main__':
    uk()
