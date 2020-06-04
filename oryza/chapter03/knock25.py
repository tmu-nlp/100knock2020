import re

def infobox(text):
    dict_info = {}
    pattern_1 = r'(?<=\}\}\t\{\{Infobox country)(.*?)(?=\}\}\t\t[A-Za-z])'
    infobox_c = re.findall(pattern_1, text)
    lines = str(infobox_c).split('\\t| ')
    for line in lines[1:]:
        to_dict = line.split(' = ')
        dict_info[to_dict[0].strip()] = ' = '.join(to_dict[1:])
    return dict_info

if __name__ == '__main__':
    file = open('united_kingdom.txt','r')
    text = file.read()
    text = text.replace('\n','\t')
    print(infobox(text))
