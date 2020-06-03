from zzz.chapter04.knock30 import load_mecab
from zzz.chapter04.knock35 import count_word
from zzz.chapter04.knock36 import bar_plot

if __name__ == '__main__':
    morpheme_text = load_mecab('neko.txt.mecab')

    counter, length = count_word(morpheme_text, co_occurrence_with='猫')
    words = []
    frequencies = []
    for (word, num) in counter[:10]:
        words.append(word)
        frequencies.append(num / length)
    # print(length)
    bar_plot(words, frequencies)