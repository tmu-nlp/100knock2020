from knock30 import getdata
from collections import defaultdict

if __name__ == "__main__":
    mecab_file_path = "neko.txt.mecab"
    document = getdata(mecab_file_path)
    word_freq = defaultdict(lambda: 0)
    for sentence in document:
        for word in sentence:
            word_freq[word["base"]] += 1
    for key, value in sorted(word_freq.items(), key=lambda x:x[1], reverse=True):
        print(f"{key}: {value}")
