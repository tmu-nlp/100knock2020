# coding:utf-8
"""
25. テンプレートの抽出Permalink
記事中に含まれる「基礎情報」テンプレートのフィールド名と値を抽出し，辞書オブジェクトとして格納せよ．
"""

import pprint
import gzip
from json import loads
import re
from knock20 import read_json

def to_list(text):
    pattern = re.compile(r'^\|(.*?)\s*=\s*(.*)$', re.MULTILINE)
    UK_tem_list = pattern.findall(text)
    return UK_tem_list

def UK_tem(UK_data): #list型
    # 正規表現　https://docs.python.org/ja/3/library/re.html#module-re　
    # DOTALL 改行も.の対象に
    pattern = re.compile(r'^.*\{\{基礎情報.*?$(.*?)^\}\}$', re.MULTILINE + re.DOTALL)
    #１×１list型になっている
    UK_tem = pattern.findall(UK_data)
    return(UK_tem[0])


if __name__ == "__main__":
    file_name = "jawiki-country.json.gz"
    UK_data = read_json(file_name)
    UK_tem_dic = {}
    for key, value in to_list(UK_tem(UK_data)):
        UK_tem_dic[key] = value
    pprint.pprint(UK_tem_dic)