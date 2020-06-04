import re
from zzz.chapter03.knock20 import extract_text
from zzz.chapter03.knock25 import template

def del_mark(text_dict, pattern, func):
    for key, value in text_dict.items():
        text_dict[key] = re.sub(pattern, func, value)
    return text_dict

if __name__ == '__main__':
    filename = 'jawiki-country.json'
    text = extract_text(filename, 'イギリス')

    basic = template(text)

    pattern = r'(\'{3})(.*?)(\'{3})'                # '''強調'''
    basic = del_mark(basic, pattern, lambda x: x.group(2))
    for (key, value) in basic.items():
        print(key, value)


