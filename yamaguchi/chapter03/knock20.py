# データ処理ライブラリ「Pandas」をインポート
import pandas as pd

def uk():
    # pandasでJSON文字列を読み込む
    data = pd.read_json('jawiki-country.json.gz', lines=True)
    # 「query()」で条件を指定
    text = data.query('title=="イギリス"')['text']

    print(text)

if __name__ == '__main__':
    uk()
