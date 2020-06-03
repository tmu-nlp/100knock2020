import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontManager

from zzz.chapter04.knock30 import load_mecab
from zzz.chapter04.knock35 import count_word

def bar_plot(words, frequencies):
    # fm = FontManager()
    # mat_fonts = set(f.name for f in fm.ttflist)
    # print(mat_fonts)

    plt.rcParams['font.sans-serif'] =  'Arial Unicode MS'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(words, frequencies)
    plt.show()

if __name__ == '__main__':
    morpheme_text = load_mecab('neko.txt.mecab')

    counter, length = count_word(morpheme_text)
    words = []
    frequencies = []
    for (word, num) in counter[:10]:
        words.append(word)
        frequencies.append(num / length)
    # print(length)
    bar_plot(words, frequencies)