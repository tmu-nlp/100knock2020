from knock30 import conll_read
from knock36 import freqs_chart
import operator

def co_accurance(sentence):
    co_occur = {}
    for sent in sentence:
        for w in range(len(sent)-2):
            alice_phrase = sent[w:w+2]
            if alice_phrase[0]['text'] == 'Alice':
                if alice_phrase[1]['text'] in co_occur:
                    co_occur[alice_phrase[1]['text']] += 1
                else:
                    co_occur[alice_phrase[1]['text']] = 1
            elif alice_phrase[1]['text'] == 'Alice':
                if alice_phrase[0]['text'] in co_occur:
                    co_occur[alice_phrase[0]['text']] += 1
                else:
                    co_occur[alice_phrase[0]['text']] = 1
    return co_occur

if __name__ == "__main__":
    texts = conll_read()
    counts = co_accurance(texts)
    freqs_chart(counts, 10)

    # for x, y in sorted(co_occur.items(), key=operator.itemgetter(1), reverse=True):
    #     print ('%s: %r' % (x, y))


        