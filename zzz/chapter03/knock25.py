import re
from zzz.chapter03.knock20 import extract_text


def parsing(text, keyword, bracket='{}'):
    res = []
    num_bracket = 0
    bracket_f = bracket[0]
    bracket_b = bracket[-1]
    start = False

    for line in text:
        if start:
            if num_bracket == 0:
                return res
            else:
                num_bracket += line.count(bracket_f)
                num_bracket -= line.count(bracket_b)
                if '*' in line:
                    # not a new item
                    res[-1] = res[-1] + ' ' + line
                else:
                    res.append(line)
        if keyword in line:
            num_bracket += 2
            start = True
            if '*' in line:
                # not a new item
                res[-1] = res[-1] + ' ' + line
            else:
                res.append(line)

    return res


def template(text, keyword):
    text = parsing(text, keyword)

    pattern = r'\|(.*?) *= *(.*)'           # |他元首等肩書1 = [[貴族院 (イギリス)|貴族院議長]]
    basic = re.findall(pattern, '\n'.join(text))

    res = {}
    for item in basic:
        res[item[0]] = item[1]
    return res


if __name__ == '__main__':
    '''
    {{title
    |key1 = value1
    |key2 = value2
    ...
    }}
    '''

    filename = 'jawiki-country.json'
    keyword = '基礎情報'
    text = extract_text(filename, 'イギリス')

    basic = template(text, keyword)
    for (key, value) in basic.items():
        print(key, value)