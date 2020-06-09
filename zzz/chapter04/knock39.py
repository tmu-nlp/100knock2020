import numpy as np
import matplotlib.pyplot as plt

from zzz.chapter04.knock30 import load_mecab
from zzz.chapter04.knock35 import count_word


if __name__ == '__main__':
    morpheme_text = load_mecab('neko.txt.mecab')

    counter, length = count_word(morpheme_text)
    counter = np.array(counter)[:, 1].astype(int)

    x = np.array([i for i in range(1, len(counter) + 1)])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.loglog(x, counter)
    plt.show()