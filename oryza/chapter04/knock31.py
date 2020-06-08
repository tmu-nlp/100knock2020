from knock30 import conll_read

def extract_postag(sentence, type, pos_tag):
    res = []
    for sent in sentence:
        for token in sent:
            if token['pos'] == pos_tag:
                res.append(token[type])
    return res

if __name__ == "__main__":
    print(extract_postag(conll_read(), 'text', 'VB'))