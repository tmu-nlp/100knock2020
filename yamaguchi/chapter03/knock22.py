# データ処理ライブラリ「Pandas」をインポート
import pandas as pd

def uk():
    # pandasでJSON文字列を読み込む
    data = pd.read_json('jawiki-country.json.gz', lines=True)

    # 「query()」で条件を指定
    text = data.query('title=="イギリス"')['text'].values[0]

    # 文章を改行で分割
    split = text.split('\n')

    # 「Category:」と書かれた行を，
    # 組み込み関数「filter()」と無名関数「lambda()」で抽出
    extraction = list(filter(lambda keyword: 'Category:' in keyword, split))

    # 不要な部分を「replace()」で削除
    result = [i.replace('[[Category:', '').replace('|*', '').replace(']]', '') for i in extraction]

    print(result)

if __name__ == '__main__':
    uk()
