import re
from zzz.chapter03.knock20 import extract_text

if __name__ == '__main__':
    filename = 'jawiki-country.json'
    text = extract_text(filename, 'イギリス')

    # pattern = r'ファイル:(.*?\..*?)\|'
    pattern = r'ファイル:(.*?\..*?)(?:\||\])'
    files = re.findall(pattern, '\n'.join(text))
    print(len(files), files)