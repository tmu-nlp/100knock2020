from collections import defaultdict
import matplotlib.pyplot as plt
import japanize_matplotlib
from knock30 import read_file

doc = read_file()

context_freq = defaultdict(lambda: 0)
window_size = 5
for morphs in doc:
    for i in range(len(morphs)):
        word = morphs[i]['surface']
        if word == '猫':
            for j in range(1, window_size+1):
                right_idx = i + j
                left_idx = i - j
                if right_idx < len(morphs):
                    context = morphs[right_idx]['surface']
                    context_freq[context] += 1
                if left_idx >= 0:
                    context = morphs[left_idx]['surface']
                    context_freq[context] += 1

# sort
sorted_context_freq = sorted(context_freq.items(), key=lambda x:x[1], reverse=True)

context_top10 = [w_f[0] for w_f in sorted_context_freq[:10]]
freq_top10 = [w_f[1] for w_f in sorted_context_freq[:10]]

plt.xlabel('context')
plt.ylabel('freq')
plt.bar(context_top10, freq_top10)
plt.show()

