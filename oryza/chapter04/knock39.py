from knock30 import conll_read
from knock35 import word_freqs
import matplotlib.pyplot as plt

texts = conll_read()
counts = word_freqs(texts)
frequency_w = []

for x, y in sorted(counts.items()):
    frequency_w.append(int(y))

count_rank = {}
for i in frequency_w:
    if i in count_rank:
        count_rank[i] += 1
    else:
        count_rank[i] = 1

rank_order = []
frequency = []

for x, y in sorted(count_rank.items()):
    rank_order.append(int(x))
    frequency.append(int(y))

fig = plt.figure()
ax = plt.gca()
ax.scatter(rank_order,frequency)
ax.set_title('Alice log-log graph')
ax.set(xlabel='Rank Order',ylabel='Frequency')
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()
