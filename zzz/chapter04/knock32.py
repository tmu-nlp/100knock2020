from zzz.chapter04.knock30 import load_mecab
from zzz.chapter04.knock31 import find_word

if __name__ == '__main__':
    morpheme_text = load_mecab('neko.txt.mecab')
    res = find_word(morpheme_text, '動詞', 'base')

    print(res)