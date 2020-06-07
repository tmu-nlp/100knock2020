import pickle
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import japanize_matplotlib

fp = open('./word_freq.pkl', 'rb')
sorted_word_freq = pickle.load(fp)

freqs = [w_f[1] for w_f in sorted_word_freq]

plt.xlabel('freq')
plt.ylabel('freq_count')
plt.hist(freqs, range=(1, 20))
plt.show()

