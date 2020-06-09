import matplotlib.pyplot as plt
import operator
from knock35 import word_freqs 
from knock30 import conll_read

def freqs_chart(count_freq, top):
    words = []
    freqs = []

    for x, y in sorted(count_freq.items(), key=operator.itemgetter(1), reverse=True)[:top]:
        words.append(str(x))
        freqs.append(int(y))

    x = range(1,top + 1)
    y = freqs

    plt.bar(x,y)
    plt.xticks(x,words)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    texts = conll_read()
    counts = word_freqs(texts)
    freqs_chart(counts, 10)