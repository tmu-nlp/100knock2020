import requests
import re

f = open('britain.txt', 'r')
britain = f.readline()
basicInfo = re.search(r'\{\{基礎情報(.*?)\\n\}\}\\n', britain).group().replace("\'\'\'", "").replace("\'\'", "")
basicInfo2 = re.sub(r'\<ref\>(.*?)\<\/ref\>', " ", basicInfo).split("\\n")

dict_basic = {}

for i in range(len(basicInfo2)):
    info = basicInfo2[i].split("=")
    if len(info) == 2:
        if re.search(r'\{\{(.*?)\|(.*?)\}\}', info[1]):
            bracket = re.search(r'\{\{(.*?)\|[a-z]{2}\|(.*?)\}\}', info[1])
            info[1] = bracket.group(2)

        elif  re.search(r'\[\[(.*?)\|(.*?)\]\]', info[1]):
            if re.search(r'(\s|\w)*\]\]', info[1]):
                info[1] = (re.search(r'(\s|\w)*\]\]', info[1]).group()[:-2])

        info[1] = info[1].replace("]]", "").replace("[[", "").replace("{{0}}", "")
        info[1] = re.sub(r'\<(.*?)\/\>', "", info[1])
        dict_basic[info[0][1:]] = info[1]

flagimage = dict_basic["国旗画像 "].replace(" ", "_")

"""
    get_imageinfo.py

    MediaWiki API Demos
    Demo of `Imageinfo` module: Get information about an image file.

    MIT License
"""

S = requests.Session()

URL = "https://en.wikipedia.org/w/api.php"

PARAMS = {
    "action": "query",
    "format": "json",
    "prop": "imageinfo",
    "titles": "File:"+flagimage,
    "iiprop": "url"
}

R = S.get(url=URL, params=PARAMS)
DATA = R.json()

PAGES = DATA["query"]["pages"]

for k, v in PAGES.items():
    print(v["title"] + "'s url is " + v["imageinfo"][0]["url"])