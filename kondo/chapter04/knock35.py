from knock30 import data_mapping
from collections import Counter

analysis_file = "neko.txt.mecab"



def word_count():
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
    print(word_count().most_common(10))
