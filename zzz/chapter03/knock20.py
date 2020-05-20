import json

def extract_text(filename, title):
    with open(filename, 'r') as file:
        # json_file = json.loads(file)
        for line in file:
            dict_line = json.loads(line)
            if dict_line['title'] == title:
                return dict_line['text'].split('\n')

if __name__ == '__main__':
    filename = 'jawiki-country.json'
    text = extract_text(filename, 'イギリス')
    print('\n'.join(text))