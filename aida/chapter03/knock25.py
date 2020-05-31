import re
from knock20 import read_gzip

def obtain_template(doc, template):
    lines, fg = [], False
    p1 = re.compile('\{\{' + template)
    p2 = re.compile('\}\}')
    p3 = re.compile('\|')
    p4 = re.compile('<ref(\s|>).+?(</ref>|$)')
    for line in doc.split('\n'):
        if fg:
            end_of_template = p2.match(line)
            is_field = p3.match(line)
            if end_of_template:
                break
            if is_field:
                lines.append(p4.sub('', line.strip()))
        if p1.match(line):
            fg = True
    p = re.compile('\|(.+?)\s=[\s]?(.+)')
    field_dic = {m.group(1): m.group(2) for m in [p.match(c) for c in lines]}

    return field_dic

if __name__ == '__main__':
    uk_doc = read_gzip()
    template = '基礎情報'
    answer = obtain_template(uk_doc, template)
    for name, value in answer.items():
        print(f'{name}: {value}')

