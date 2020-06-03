import re
from knock25 import infobox
from knock26 import rmv_emphasis
from knock27 import rmv_int_links


def rmv_markups(dict_info):
    for key, value in dict_info.items():
        dict_info[key] = re.sub(r'\<.*?\>.*?(\<\/.*?\>)',r' ',value) # remove <ref>  
        dict_info[key] = re.sub(r'(\[\[)(.*?)(\]\])', lambda x: x.group(2), dict_info[key]) # remove [[.*?]] or {{.*?}}
        dict_info[key] = re.sub(r'(\{\{)(.*?)(\}\})', lambda x: x.group(2), dict_info[key]) # remove [[.*?]] or {{.*?}}
        dict_info[key] = re.sub(r'\<.*?\>',r' ', dict_info[key]) # remove <.*?>  
        dict_info[key] = re.sub(r'&nbsp;',r' ', dict_info[key]) # remove &nbsp;
    return dict_info

if __name__ == '__main__':
    file = open('united_kingdom.txt','r')
    text = file.read()
    text = text.replace('\n','\t')
    
    infobox_d = infobox(text)
    infobox_d = rmv_emphasis(infobox_d)
    infobox_d = rmv_int_links(infobox_d)
    infobox_d = rmv_markups(infobox_d) 
    # print(infobox_d)

    with open('uk_plain.txt','w') as outf:
        for key, value in infobox_d.items():
            outf.write(key + ' : ' + infobox_d[key] + '\n')