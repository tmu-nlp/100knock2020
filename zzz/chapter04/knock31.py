from zzz.chapter04.knock30 import load_mecab


def find_word(morpheme_text, pos, target):
    res = []
    for sentence in morpheme_text:
        for word in sentence:
            if word['pos'] == pos:
                res.append(word[target])
    return res


if __name__ == '__main__':
    morpheme_text = load_mecab('neko.txt.mecab')

    res = find_word(morpheme_text, '動詞', 'surface')
    print(res)
