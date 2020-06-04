# coding:utf-8
"""
23. セクション構造
記事中に含まれるセクション名とそのレベル（例えば”== セクション名 ==”なら1）を表示せよ．
"""

import gzip
from json import loads
import re
from knock20 import read_json

def UK_sec(UK_data): #list型
    # 正規表現　https://docs.python.org/ja/3/library/re.html#module-re　
    # カーリングと自転車競技に余分なスペースあり
    pattern = re.compile(r'^(={2,})\s*(.+?)\s*\1.*$', re.MULTILINE)
    UK_sec_list = pattern.findall(UK_data)
    return UK_sec_list

if __name__ == "__main__":
    file_name = "jawiki-country.json.gz"
    UK_data = read_json(file_name)
    for line in UK_sec(UK_data):
        level = len(line[0]) - 1
        print("{}({})".format(line[1], level))