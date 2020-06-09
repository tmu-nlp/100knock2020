from knock30 import conll_read

def aofb_long(sentence):
    seqs = []
    seq = []
    for sent in sentence:
        for w in sent:
            if w['pos'] == 'NN':
                seq.append(w['text'])
            else:
                if len(seq) > 1:
                    seqs.append(seq)
                seq = []
    return seqs

if __name__ == "__main__":
    phrase = aofb_long(conll_read())
    for w in phrase:
        print(' '.join(w) + '\n')