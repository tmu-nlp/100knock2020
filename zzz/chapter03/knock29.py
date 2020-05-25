import re
from zzz.chapter03.knock20 import extract_text
from zzz.chapter03.knock25 import template
from zzz.chapter03.knock26 import del_mark


if __name__ == '__main__':
    filename = 'jawiki-country.json'
    text = extract_text(filename, 'イギリス')

    basic = template(text)

    pattern = r'(\'{3})(.*?)(\'{3})'
    basic = del_mark(basic, pattern, lambda x: x.group(2))
    pattern = r'(\[\[)(.*?)(\]\])'
    basic = del_mark(basic, pattern, lambda x: x.group(2))
    pattern = r'(\'{2})(.*?)(\'{2})'
    basic = del_mark(basic, pattern, lambda x: x.group(2))
    pattern = r'(\'{2})(.*?)(\'{2})'
    basic = del_mark(basic, pattern, lambda x: x.group(2))
    pattern = r'(\[)(.*?)(\])'
    basic = del_mark(basic, pattern, lambda x: x.group(2))

    print(basic['国旗画像'])
    import requests

    url = "https://www.mediawiki.org/w/api.php"

    paras = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "titles": 'file:' + basic['国旗画像'],
        "iiprop": "url",
    }

    r = requests.get(url, paras)
    data = r.json()
    pages = data["query"]["pages"]['-1']

    imageinfo = pages['imageinfo'][0]
    # print(imageinfo)
    print(imageinfo['url'])