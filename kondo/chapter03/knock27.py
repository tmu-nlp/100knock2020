# coding:utf-8
"""
27. 内部リンクの除去
26の処理に加えて，テンプレートの値からMediaWikiの内部リンクマークアップを除去し，テキストに変換せよ（参考: マークアップ早見表）．
"""

import gzip
from json import loads
import re
from knock20 import read_json
from knock25 import UK_tem

def del_lin(text): #list型
    # 正規表現　https://docs.python.org/ja/3/library/re.html#module-re　
    pattern = re.compile(r'(\'{2,5})(.*?)(\1)', re.MULTILINE)
    del_mar_tex = pattern.sub(r'\2', text)
    # (?!...) 否定先読みアサーション
    pattern = re.compile(r'(?!#REDIRECT)(?:\[\[)(?!ファイル)(.*?)(?:\|.*?)??(?:\]\])', re.MULTILINE)
    del_lin_tex = pattern.sub(r'\1', del_mar_tex)
    return del_lin_tex

if __name__ == "__main__":
    file_name = "jawiki-country.json.gz"
    UK_data = read_json(file_name)
    UK_tem_list = UK_tem(UK_data)
    print(del_lin(UK_tem_list))