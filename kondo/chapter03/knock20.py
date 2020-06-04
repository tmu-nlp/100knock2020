# coding:utf-8
"""
20. JSONデータの読み込み
Wikipedia記事のJSONファイルを読み込み，「イギリス」に関する記事本文を表示せよ．
問題21-29では，ここで抽出した記事本文に対して実行せよ．
"""

import gzip
import json

def read_json(file_path): #str型
    #rt→テキストファイルで読み込む
    with gzip.open(file_path, 'rt') as open_file:
        for line in open_file:
            #json文字列を辞書に変換
            file_data = json.loads(line)
            if file_data['title'] == 'イギリス':
                return(file_data['text'])

if __name__ == "__main__":
    file_name = "jawiki-country.json.gz"
    print(read_json(file_name))

