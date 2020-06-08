from knock30 import conll_read
import operator

def word_freqs(sentence):
    word_count = {}
    for sent in sentence:
        for w in sent:
            if w['text'] in word_count:
                word_count[w['text']] += 1
            else:
                word_count[w['text']] = 1
    return word_count

if __name__ == "__main__":
    text = conll_read()
    counts = word_freqs(text)

    for x, y in sorted(counts.items(), key=operator.itemgetter(1), reverse=True):
        print ('%s: %r' % (x, y))