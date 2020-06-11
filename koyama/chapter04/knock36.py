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
    x = []
    y = []
    for key, value in sorted(word_freq.items(), key=lambda x:x[1], reverse=True):
        x.append(key)
        y.append(value)
        if len(x) >= 10:
            break
    fp = FontProperties(fname=r"/Users/aomi/Library/Fonts/ipaexg.ttf")
    plt.bar(x, y)
    plt.xticks(fontproperties=fp)
    plt.xlabel('出現頻度が高い10語', fontproperties=fp)
    plt.show()