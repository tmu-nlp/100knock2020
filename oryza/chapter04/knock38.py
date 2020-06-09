from knock30 import conll_read
from knock35 import word_freqs
import matplotlib.pyplot as plt
import operator

texts = conll_read()
counts = word_freqs(texts)
freqs = []

for x, y in sorted(counts.items()):
    freqs.append(int(y))

plt.hist(freqs, bins = 100)
plt.title('Alice Histogram of Word Frequency')
plt.xlabel('Word Frequency')
plt.ylabel('Number of Unique Words')
plt.show()

# c = {}
# for i in freqs:
#     if i in c:
#         c[i] += 1
#     else:
#         c[i] = 1

# for x, y in sorted(c.items(), key=operator.itemgetter(1), reverse=True):
#     print ('%s: %r' % (x, y))