import re
from knock20 import read_gzip
from knock25 import obtain_template 
from knock26 import remove_emphasis
from knock27 import remove_internallink, postprocess

def remove_externallink(field_dic):
    removed_internallink_dic = {}
    for key, value in field_dic.items():
        removed_internallink_dic[key] = re.sub(r"\[\[([^|#\]]+?\|)*(.*?)\]\]", r"\2", value)
    
    return removed_internallink_dic


def remove_tokens(field_dic):
    removed_tokens_dic = {}
    for key, value in field_dic.items():
        value = re.sub(r"<\/?[br|ref][^>]+?\/?>", "", value)
        value = re.sub(r"\{\{lang\|[^\|]+?\|([^\}]+?)\}\}", r"\1", value)
        removed_tokens_dic[key] = value

    return removed_tokens_dic


if __name__ == '__main__':
    uk_doc = read_gzip()
    template = '基礎情報'
    field_dic = obtain_template(uk_doc, template)
    methods = [remove_emphasis, remove_internallink, remove_externallink, remove_tokens]
    answer = postprocess(field_dic, methods)
    for name, value in answer.items():
        print(f'{name}: {value}')

