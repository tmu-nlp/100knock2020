from zzz.chapter04.knock30 import load_mecab
from collections import Counter

def count_word(morpheme_text, co_occurrence_with=None):
    cache = []
    length = 0
    if co_occurrence_with is None:
        for sentence in morpheme_text:
            for word in sentence:
                cache.append(word['base'])
        length = len(cache)
    else:
        for sentence in morpheme_text:
            length += len(sentence)
            # if co_occurrence_with in [word['base'] for word in sentence]:
            #     for word in sentence:
            #         if word['base'] != co_occurrence_with:
            #             cache.append(word['base'])
            for index in range(len(sentence) - 1):
                if sentence[index]['base'] == co_occurrence_with:
                    cache.append(sentence[index + 1]['base'])
                if sentence[index + 1]['base'] == co_occurrence_with:
                    cache.append(sentence[index]['base'])

    counter = Counter(cache)
    counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    return counter, length


if __name__ == '__main__':
    morpheme_text = load_mecab('neko.txt.mecab')

    counter, length = count_word(morpheme_text)
    for (word, num) in counter:
        print(word, ':', float(num) / length)
