import pickle
from collections import Counter
import matplotlib.pyplot as plt
import japanize_matplotlib

fp = open('./word_freq.pkl', 'rb')
sorted_word_freq = pickle.load(fp)

words_top10 = [w_f[0] for w_f in sorted_word_freq[:10]]
freq_top10 = [w_f[1] for w_f in sorted_word_freq[:10]]

plt.xlabel('word')
plt.ylabel('freq')
plt.bar(words_top10, freq_top10)
plt.show()

