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
    print(basic)

    pattern = r'(\[\[)(.*?)(\]\])'
    basic = del_mark(basic, pattern, lambda x: x.group(2))
    print(basic)

    #remove ltalics
    pattern = r'(\'{2})(.*?)(\'{2})'
    basic = del_mark(basic, pattern, lambda x: x.group(2))
    print(basic)

    #remove ltalics
    pattern = r'(\'{2})(.*?)(\'{2})'
    basic = del_mark(basic, pattern, lambda x: x.group(2))
    print(basic)

    #remove outter link
    pattern = r'(\[)(.*?)(\])'
    basic = del_mark(basic, pattern, lambda x: x.group(2))
    print(basic)