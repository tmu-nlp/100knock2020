# coding:utf-8
"""
26. 強調マークアップの除去
25の処理時に，テンプレートの値からMediaWikiの強調マークアップ（弱い強調，強調，強い強調のすべて）を除去してテキストに変換せよ．
"""

import gzip
from json import loads
import re
from knock20 import read_json
from knock25 import UK_tem

def del_mar(text): #list型
    # 正規表現　https://docs.python.org/ja/3/library/re.html#module-re　
    pattern = re.compile(r'(\'{2,5})(.*?)(\1)', re.MULTILINE)
    del_mar_tex = pattern.sub(r'\2', text)
    return del_mar_tex

if __name__ == "__main__":
    file_name = "jawiki-country.json.gz"
    UK_data = read_json(file_name)
    UK_tem_list = UK_tem(UK_data)
    print(del_mar(UK_tem_list))