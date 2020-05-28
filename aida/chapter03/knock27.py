import re
from knock20 import read_gzip
from knock25 import obtain_template 
from knock26 import remove_emphasis

def remove_internallink(field_dic):
    removed_internallink_dic = {}
    for key, value in field_dic.items():
        removed_internallink_dic[key] = re.sub(r"\[\[([^|#\]]+?\|)*(.*?)\]\]", r"\2", value)
    
    return removed_internallink_dic


def postprocess(field_dic, methods=[]):
    for method in methods:
        field_dic = method(field_dic)
    return field_dic


if __name__ == '__main__':
    uk_doc = read_gzip()
    template = '基礎情報'
    field_dic = obtain_template(uk_doc, template)
    methods = [remove_emphasis, remove_internallink]
    answer = postprocess(field_dic, methods)
    for name, value in answer.items():
        print(f'{name}: {value}')

