from zzz.chapter04.knock30 import load_mecab


if __name__ == '__main__':
    morpheme_text = load_mecab('neko.txt.mecab')
    noun_phrase = []
    for sentence in morpheme_text:

        temp_phrase = []
        for word in sentence:
            if word['pos'] == '名詞':
                temp_phrase.append(word['surface'])
            else:
                if len(temp_phrase) > 0:
                    noun_phrase.append(''.join(temp_phrase))
                temp_phrase = []
        if len(temp_phrase) > 0:
            noun_phrase.append(''.join(temp_phrase))

    print(noun_phrase)