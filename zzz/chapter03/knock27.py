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
    # print(basic)

    # pattern = r'(\[\[.*?)(#\|.*?)(\]\])'
    # basic = del_mark(basic, pattern, 2)
    # print(basic)
    pattern = r'(\[\[)(.*?)(:?#.*?|)(\|)(.*?)(\]\])'                          # [[記事名|表示文字]], [[記事名#節名|表示文字]]
    basic = del_mark(basic, pattern, lambda x: x.group(5))
    pattern = r'(\[\[)(.*?)(\]\])'                                              # [[記事名]]
    basic = del_mark(basic, pattern, lambda x: x.group(2))

    for (key, value) in basic.items():
        print(key, value)
