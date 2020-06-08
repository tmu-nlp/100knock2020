def conll_read():
    with open('./alice/alice.txt.conll','r') as f:
        sentences = []
        words = []
        for line in f:
            if line == '\n':
                if len(words) > 0:
                    sentences.append(words)
                words = []
            else:
                lines = line.split('\t')
                dic = {
                    'id': lines[0],
                    'text': lines[1],
                    'lemma': lines[2],
                    'pos': lines[3]
                }
                words.append(dic)
    return sentences

if __name__ == "__main__":
    print(conll_read())