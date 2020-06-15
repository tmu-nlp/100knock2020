import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from knock30 import data_mapping
from collections import Counter

analysis_file = "neko.txt.mecab"

def describe_top10(table)->"None":
    sample = []
    data = []
    for i in range(10):
        sample.append(table[i][0])
        data.append(table[i][1])
    left = np.array(sample)
    height = np.array(data)
    plt.bar(left, height)
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
    #上位10個
    describe_top10(word_count().most_common(10))