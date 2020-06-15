import pickle
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import japanize_matplotlib

fp = open('./word_freq.pkl', 'rb')
sorted_word_freq = pickle.load(fp)

counts = [f_c[1] for f_c in sorted_word_freq]

plt.xlabel('順位')
plt.ylabel('頻度')
plt.xscale('log')
plt.yscale('log')
plt.plot(range(1, len(counts)+1), counts)
plt.show()

