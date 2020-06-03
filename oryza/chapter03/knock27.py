import re
from knock25 import infobox
from knock26 import rmv_emphasis

def rmv_int_links(dict_info):
    for key, value in dict_info.items():
        dict_info[key] = re.sub(r'(\<.*https?:\/\/.*?[\r]*\<\/.*?\>)', '', value)
    return dict_info

if __name__ == '__main__':
    file = open('united_kingdom.txt','r')
    text = file.read()
    text = text.replace('\n','\t')
    
    infobox_d = infobox(text)
    infobox_d = rmv_emphasis(infobox_d)
    infobox_d = rmv_int_links(infobox_d)
    # print(infobox_d)

    for key, value in infobox_d.items():
        print(infobox_d[key])