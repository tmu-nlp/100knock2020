#https://note.nkmk.me/python-re-match-search-findall-etc/
#https://note.nkmk.me/python-raw-string-escape/
'''
21. カテゴリ名を含む行を抽出Permalink
記事中でカテゴリ名を宣言している行を抽出せよ．
'''
import json
import re

with open('jawiki-country.json') as file:
    for line in file:
        line_dict = json.loads(line) # load():ファイルを読み込む, loads():str型を読み込む
        if line_dict['title'] == 'イギリス':
            text = line_dict['text']
        else: pass
    #行ごとに分ける
    text_list = text.split('\n')

#メタ文字 ". ^ $ * + ? { } [ ] \ | ( )"
#https://docs.python.org/ja/3/howto/regex.html
    for line in text_list:
        #マッチオブジェクトはブール値として判定される場合は常にTrueとして扱われる。
        #match()とsearch()ではマッチしない場合はNoneを返す。Noneはブール値としてはFalseと判定される。
        if re.match(r'^\[\[Category:.+\]\]$', line):
            print(line)
        else: pass
