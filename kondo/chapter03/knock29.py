# coding:utf-8
"""
29. 国旗画像のURLを取得するPermalink
テンプレートの内容を利用し，国旗画像のURLを取得せよ．（ヒント: MediaWiki APIのimageinfoを呼び出して，ファイル参照をURLに変換すればよい）．
"""

import gzip
from json import loads
import re
import requests
from knock20 import read_json
from knock25 import UK_tem
from knock28 import del_lin

def flag(data):
    pattern = re.compile(r'^\|国旗画像\s*=\s*(.*)$', re.MULTILINE)
    flag_data = pattern.findall(data)
    return flag_data[0]

def get_URL(svg_data):
    url = "https://www.mediawiki.org/w/api.php"
    params  =  {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "iiprop": "url",
        "titles": "File:" + svg_data,
    }

    R = requests.get(url, params).json()
    return R["query"]["pages"]["-1"]["imageinfo"][0]["url"]

if __name__ == "__main__":
    file_name = "jawiki-country.json.gz"
    UK_data = read_json(file_name)
    UK_tem_list = UK_tem(UK_data)
    UK_data = del_lin(UK_tem_list)
    UK_flag = flag(UK_data)
    UJ_URL = get_URL(UK_flag)
    print(UJ_URL)