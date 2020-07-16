from collections import defaultdict
import string

def get_feature(data):
    data = data.readlines()
    text = []
    for line in data:
        text.append(line.strip().split('\t')[0])
    
    return text

def get_label(data):
    data = data.readlines()
    text = []
    for line in data:
        text.append(line.strip().split('\t')[1])
    
    return text

def word2ids(text, unk=0):
    d = defaultdict(int)
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    for line in text:
        line = line.strip()
        for word in line.translate(table).split():
            d[word] += 1
    d = sorted(d.items(), key=lambda x:x[1], reverse=True)

    word2id = {word: i + 1 for i, (word, cnt) in enumerate(d) if cnt > 1}

    text2id = []
    for line in text:
        line = line.strip()
        text2id.append([word2id.get(word, unk) for word in line.translate(table).split()])  
    
    return text2id, word2id

if __name__ == "__main__":
    text = get_feature(open('test2.feature.txt'))
    text2ids = word2ids(text)
    print(text2ids)

    # text: week workers least little
    # ids: [45, 817, 882, 314]
    