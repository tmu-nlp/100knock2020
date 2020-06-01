from zzz.chapter04.knock30 import load_mecab


if __name__ == '__main__':
    morpheme_text = load_mecab('neko.txt.mecab')
    target = {'surface': 'の', 'pos': '助詞', 'pos1': '連体化','base': 'の'}
    res = []
    for sentence in morpheme_text:
        if target in sentence:
            index = sentence.index(target)
            if 0 < index < len(sentence) - 1:
                res.append(''.join([sentence[index + bias]['surface'] for bias in [-1, 0, 1]]))
    print(res)
