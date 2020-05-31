import MeCab
import re

def write_to_mecab():
    with open('neko.txt', 'r') as file :
        with open('neko.txt.mecab', 'w') as o_file:
            mecab = MeCab.Tagger()
            for line in file:
                res = mecab.parse(line)
                # print(res)
                o_file.write(res)


def load_mecab(filename):
    types = ['surface', 'pos', 'pos1', '_', '_', '_', '_', 'base', '_', '_']
    morpheme_text = []
    morpheme_sent = []
    with open(filename, 'r') as file:
        text = ''.join([line for line in file]).split('\n')
        for word in text:
            if len(word) > 0:
                if word == 'EOS':
                    morpheme_text.append(morpheme_sent)
                    morpheme_sent = []
                else:
                    item = re.split(r'\t|,|\n', word)
                    if len(item) == len(types):
                        morpheme_word = {key: value for (key, value) in zip(types, item)}
                        morpheme_word.pop('_')
                    # print(morpheme)
                    morpheme_sent.append(morpheme_word)
    return morpheme_text



if __name__ == '__main__':
    # write_to_mecab()
    morpheme_text = load_mecab('neko.txt.mecab')
    # for index, sent in enumerate(morpheme_text):
    #     if index == 5:
    #         break
    #     print(sent)
