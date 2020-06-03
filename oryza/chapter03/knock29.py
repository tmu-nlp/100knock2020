import re
import requests
from knock25 import infobox
from knock26 import rmv_emphasis
from knock27 import rmv_int_links
from knock28 import rmv_markups

if __name__ == '__main__':
    file = open('united_kingdom.txt','r')
    text = file.read()
    text = text.replace('\n','\t')
    
    infobox_d = infobox(text)
    infobox_d = rmv_emphasis(infobox_d)
    infobox_d = rmv_int_links(infobox_d)
    infobox_d = rmv_markups(infobox_d) 

    print(infobox_d['image_flag'])

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
        "titles": "File:" + infobox_d['image_flag'],
        "iiprop": "url"
    }

    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()

    PAGES = DATA["query"]["pages"]
    page = PAGES['23473560']
    image_info = page['imageinfo']

    print(image_info[0]['url'])