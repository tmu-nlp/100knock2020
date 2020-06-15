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
    y = []
    for key, value in sorted(word_freq.items(), key=lambda x:x[1], reverse=True):
        y.append(value)
    fp = FontProperties(fname=r"/Users/aomi/Library/Fonts/ipaexg.ttf")
    plt.hist(y, bins=1000)
    plt.axis([0, 9200, 0, 5000])
    plt.xlabel('出現頻度', fontproperties=fp)
    plt.ylabel('単語の種類数', fontproperties=fp)
    plt.show()