# coding:utf-8
"""
21. カテゴリ名を含む行を抽出Permalink
記事中でカテゴリ名を宣言している行を抽出せよ．
"""

import gzip
from json import loads
import re
from knock20 import read_json

def UK_cat(UK_data): #list型
    # 正規表現　https://docs.python.org/ja/3/library/re.html#module-re
    pattern = re.compile(r'^\[\[Category:.*\]\]$', re.MULTILINE)
    UK_cat_list = pattern.findall(UK_data)
    return UK_cat_list

if __name__ == "__main__":
    file_name = "jawiki-country.json.gz"
    UK_data = read_json(file_name)
    for line in UK_cat(UK_data):
        print(line)