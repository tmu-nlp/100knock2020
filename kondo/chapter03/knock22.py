# coding:utf-8
"""
22. カテゴリ名の抽出
記事のカテゴリ名を（行単位ではなく名前で）抽出せよ．
"""

import gzip
from json import loads
import re
from knock20 import read_json

def UK_cat_name(UK_data): #list型
    # 正規表現　https://docs.python.org/ja/3/library/re.html#module-re ?で非貪欲(出来るだけ少ない文字数)
    pattern = re.compile(r'^\[\[Category:(.*?)(?:\|.*)?\]\]$', re.MULTILINE)
    UK_cat_list = pattern.findall(UK_data)
    return UK_cat_list

if __name__ == "__main__":
    file_name = "jawiki-country.json.gz"
    UK_data = read_json(file_name)
    for line in UK_cat_name(UK_data):
        print(line)