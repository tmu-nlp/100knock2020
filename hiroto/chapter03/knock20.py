#https://teratail.com/questions/146630
'''
20. JSONデータの読み込みPermalink
Wikipedia記事のJSONファイルを読み込み，「イギリス」に関する記事本文を表示せよ．
問題21-29では，ここで抽出した記事本文に対して実行せよ．
'''
import json

with open('jawiki-country.json') as file:
    for line in file:
        line_dict = json.loads(line) # load():ファイルを読み込む, loads():str型を読み込む
        if line_dict['title'] == 'イギリス':
            print(line_dict['text'])
        else: pass
