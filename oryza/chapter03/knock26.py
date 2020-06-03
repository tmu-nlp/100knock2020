import re
from knock25 import infobox

def rmv_emphasis(dict_info):
    for key, value in dict_info.items():
        dict_info[key] = re.sub(r'(\\\'\\\')|(\\\'\\\'\\\')', '', value)
    return dict_info

if __name__ == '__main__':
    file = open('united_kingdom.txt','r')
    text = file.read()
    text = text.replace('\n','\t')
    
    infobox_d = infobox(text)
    infobox_d = rmv_emphasis(infobox_d)
    # print(infobox_d)

    for key, value in infobox_d.items():
        print(infobox_d[key])
