import re
from knock20 import read_gzip
from knock25 import obtain_template 

def remove_emphasis(field_dic):
    removed_emphasis_dic = {}
    for key, value in field_dic.items():
        removed_emphasis_dic[key] = re.sub(r"\'+", '', value) 
    
    return removed_emphasis_dic


if __name__ == '__main__':
    uk_doc = read_gzip()
    template = '基礎情報'
    field_dic = obtain_template(uk_doc, template)
    answer = remove_emphasis(field_dic)
    for name, value in answer.items():
        print(f'{name}: {value}')

