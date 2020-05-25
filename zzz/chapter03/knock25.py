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
                    res[-1] = res[-1] + ' ' + line
                else:
                    res.append(line)
        if keyword in line:
            num_bracket += line.count(bracket_f)
            start = True
            if '*' in line:
                res[-1] = res[-1] + ' ' + line
            else:
                res.append(line)

    return res


def template(text):
    text = parsing(text, '基礎情報')

    pattern = r'\|(.*?) *= *(.*)'
    basic = re.findall(pattern, '\n'.join(text))

    res = {}
    for item in basic:
        res[item[0]] = item[1]
    return res


if __name__ == '__main__':
    filename = 'jawiki-country.json'
    text = extract_text(filename, 'イギリス')

    basic = template(text)
    print(basic)