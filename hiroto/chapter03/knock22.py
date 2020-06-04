#https://docs.python.org/ja/3/library/re.html
'''
22. カテゴリ名の抽出Permalink
記事のカテゴリ名を（行単位ではなく名前で）抽出せよ．
'''
import json
import re

with open('jawiki-country.json') as file:
    for line in file:
        line_dict = json.loads(line) # load():ファイルを読み込む, loads():str型を読み込む
        if line_dict['title'] == 'イギリス':
            text = line_dict['text']
        else: pass
#^\[\[Category:.+\]\]$
#():キャプチャする
#(?:):キャプチャしない
#findallに与える正規表現にキャプチャするカッコを与えると、全体ではなくカッコの中だけ返すようになってしまいます。
#貪欲マッチ:最も長い文字列にマッチ
#非貪欲マッチ:最も短い文字列にマッチする
    p = re.compile(r'''
        ^
        \[\[Category:
        (.+?)
        (?:\|.*)?
        \]\]
        $
        ''', re.MULTILINE + re.VERBOSE) #MULTILINE:^と$が各行の先頭と終了も考慮する, VERBOSE:間隔を無視

    #re.findall()はマッチするすべての部分を文字列のリストとして返す。
    li = p.findall(text)
    for category in li:
        print(category)
