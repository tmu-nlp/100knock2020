import pickle
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import japanize_matplotlib

fp = open('./word_freq.pkl', 'rb')
sorted_word_freq = pickle.load(fp)

freq_count = defaultdict(lambda: 0)
for w_f in sorted_word_freq:
    freq = w_f[1]
    freq_count[freq] += 1

sorted_freq_count = sorted(freq_count.items(), key=lambda x:x[1], reverse=True)
counts = [f_c[1] for f_c in sorted_freq_count]

plt.xlabel('rank')
plt.ylabel('freq_count')
plt.xscale('log')
plt.yscale('log')
plt.plot(range(1, len(counts)+1), counts)
plt.show()

