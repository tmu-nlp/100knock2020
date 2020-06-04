import re
from zzz.chapter03.knock20 import extract_text

if __name__ == '__main__':
    filename = 'jawiki-country.json'
    text = extract_text(filename, 'イギリス')

    pattern = r'=+.*?=+?\n'         # ===スポーツ===, ====サッカー====, ...
    sections = re.findall(pattern, '\n'.join(text))
    for item in sections:
        print(item.replace('\n', '').replace('=', ''), str(int(item.count('=') / 2)))
