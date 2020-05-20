from zzz.chapter03.knock20 import extract_text

if __name__ == '__main__':
    filename = 'jawiki-country.json'
    text = extract_text(filename, 'イギリス')
    print('\n'.join([line for line in text if 'Category' in line]))
