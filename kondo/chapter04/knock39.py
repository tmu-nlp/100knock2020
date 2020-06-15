import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from knock30 import data_mapping
from collections import Counter

analysis_file = "neko.txt.mecab"

def describe(table)->"None":
    val = []
    for key, value in table:
        val.append(value)
    
    plt.scatter(
        range(1, len(val) + 1),
        val
    )

    plt.xlim(1, len(val) + 1)
    plt.ylim(1, val[0])

    plt.xscale('log')
    plt.yscale('log')

    plt.show()

def word_count()->list:
    lines = data_mapping(analysis_file)
    longest_N_list = []
    word_list = []
    for line in lines:
        for word in line:
            word_list.append(word['surface'])
    return(Counter(word_list))

if __name__ == "__main__":
    t = 0
    dic = word_count()
    re = []
    for key in dic:
        t += 1
    describe(dic.most_common(t))