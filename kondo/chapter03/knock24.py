# coding:utf-8
"""
24. ファイル参照の抽出
記事から参照されているメディアファイルをすべて抜き出せ．
"""

import gzip
from json import loads
import re
from knock20 import read_json

def UK_cit(UK_data): #list型
    # 正規表現　https://docs.python.org/ja/3/library/re.html#module-re　
    pattern = re.compile(r'^\[\[ファイル:(.*?)\|.*\]\]$', re.MULTILINE)
    UK_sec_list = pattern.findall(UK_data)
    return UK_sec_list

if __name__ == "__main__":
    file_name = "jawiki-country.json.gz"
    UK_data = read_json(file_name)
    for line in UK_cit(UK_data):
        print(line)