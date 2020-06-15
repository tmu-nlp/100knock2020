import MeCab
with open('neko.txt') as r_file\
    , open('neko.txt.mecab', 'w') as w_file:
    mecab = MeCab.Tagger()
    w_file.write(mecab.parse(r_file.read()))
