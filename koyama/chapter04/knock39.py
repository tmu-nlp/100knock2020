from knock30 import getdata
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

if __name__ == "__main__":
    mecab_file_path = "neko.txt.mecab"
    document = getdata(mecab_file_path)
    word_freq = defaultdict(lambda: 0)
    for sentence in document:
        for word in sentence:
            word_freq[word["base"]] += 1
    counts = []
    for key, value in sorted(word_freq.items(), key=lambda x:x[1], reverse=True):
        counts.append(value)
    fp = FontProperties(fname=r"/Users/aomi/Library/Fonts/ipaexg.ttf")
    plt.scatter(range(1, len(counts) + 1), counts)
    plt.xlim(1, len(counts) + 1)
    plt.ylim(1, counts[0])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('出現頻度順位', fontproperties=fp)
    plt.ylabel('出現頻度', fontproperties=fp)
    plt.show()