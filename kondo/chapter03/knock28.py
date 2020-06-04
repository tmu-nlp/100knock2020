# coding:utf-8
"""
28. MediaWikiマークアップの除去
27の処理に加えて，テンプレートの値からMediaWikiマークアップを可能な限り除去し，国の基本情報を整形せよ．
"""

import gzip
from json import loads
import re
from knock20 import read_json
from knock25 import UK_tem

def del_lin(text): #str型
    # 正規表現　https://docs.python.org/ja/3/library/re.html#module-re　
    pattern = re.compile(r'(\'{2,5})(.*?)(\1)', re.MULTILINE)
    del_tex = pattern.sub(r'\2', text)
    # [[]]系削除
    pattern = re.compile(r'(?:\[\[)(.*?)(?:\|.*?)??(?:\]\])', re.MULTILINE)
    del_tex = pattern.sub(r'\1', del_tex)
    #<br />削除
    pattern = re.compile(r'(?:\<br \/\>)', re.MULTILINE)
    del_tex = pattern.sub(r'', del_tex)
    #{{lang|...|...}}
    pattern = re.compile(r'(.*?)(?:\{\{lang\|)(?:.*?\|)(.*?)(?:\|.*?)??(?:\}\})(.*?)', re.MULTILINE)
    del_tex = pattern.sub(r'\1\2\3', del_tex)
    #{{仮リンク|...|...}}
    pattern = re.compile(r'(.*?)(?:\{\{仮リンク\|)(.*?)(?:\|.*?)??(?:\}\})(.*?)', re.MULTILINE)
    del_tex = pattern.sub(r'\1\2\3', del_tex)
    #{{en icon}}
    pattern = re.compile(r'(\{\{en icon\}\})', re.MULTILINE)
    del_tex = pattern.sub(r',', del_tex)
    #殘りの{{}}削除
    pattern = re.compile(r'(\{\{.*?\}\})', re.MULTILINE)
    del_tex = pattern.sub(r'', del_tex)
    #<ref name ... />
    pattern = re.compile(r'(\<ref name.*?)(\/\>)', re.MULTILINE)
    del_tex = pattern.sub(r'', del_tex)
    #<ref>...</ref>
    pattern = re.compile(r'(?:\<ref\>)(.*?)(?:\<\/ref\>)', re.MULTILINE)
    del_tex = pattern.sub(r'\1', del_tex)
    #<ref name...>...</ref>
    pattern = re.compile(r'(?:\<ref name.*?\<\/ref\>)', re.MULTILINE)
    del_tex = pattern.sub(r'', del_tex)
    #<ref>or</ref>
    pattern = re.compile(r'(?:\<ref\>|\<\/ref\>)', re.MULTILINE)
    del_tex = pattern.sub(r'', del_tex)
    return del_tex

if __name__ == "__main__":
    file_name = "jawiki-country.json.gz"
    UK_data = read_json(file_name)
    UK_tem_list = UK_tem(UK_data)
    print(del_lin(UK_tem_list))