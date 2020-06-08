from knock30 import conll_read
from knock31 import extract_postag

def extract_aofb(sentence):
    res = []
    for sent in sentence:
        for w in range(len(sent)-3):
            phrase = sent[w:w+3]
            w1 = phrase[0]['pos'] == 'NN'
            w2 = phrase[1]['text'] == 'of'
            w3 = phrase[2]['pos'] == 'NN'
            if w1 and w2 and w3:
                res.append(word['text'] for word in phrase)
    return res

if __name__ == "__main__":
    phrase = extract_aofb(conll_read())
    for w in phrase:
        print(' '.join(w) + '\n')