import re
import requests

from knock20 import read_gzip
from knock25 import obtain_template 
from knock26 import remove_emphasis
from knock27 import remove_internallink, postprocess
from knock28 import remove_externallink, remove_tokens


def get_url(field_dic):
    url_file = field_dic['国旗画像'].replace(' ', '_')
    url = 'https://commons.wikimedia.org/w/api.php?action=query&titles=File:' + url_file + '&prop=imageinfo&iiprop=url&format=json'
    data = requests.get(url)
    return re.search(f'"url":"(.+?)"', data.text).group(1)
    

if __name__ == '__main__':
    uk_doc = read_gzip()
    template = '基礎情報'
    field_dic = obtain_template(uk_doc, template)
    methods = [remove_emphasis, remove_internallink, remove_externallink, remove_tokens]
    field_dic = postprocess(field_dic, methods)
    answer = get_url(field_dic)
    print(answer)

