import pickle
from collections import Counter
from knock30 import read_file

doc = read_file()
word_freq = Counter()
for morphs in doc:
    for morph in morphs:
        word = morph['surface']
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

# sort
sorted_word_freq = sorted(word_freq.items(), key=lambda x:x[1], reverse=True)
for w_f in sorted_word_freq:
    word = w_f[0]
    freq = w_f[1]
    print('{}\t{}'.format(word, freq))

fp = open('word_freq.pkl', 'wb')
pickle.dump(sorted_word_freq, fp)

