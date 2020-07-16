import pickle
from collections import defaultdict

def obtain_data(file, is_train=False):
    """ obtain labels and features from data file
    
    :param word_freq: defaultdict, word frequency
    :param X_data: list, words
    :param y_data: list, label nums
    :return: X_data, y_data, word_freq
    """
    if is_train:
        word_freq = defaultdict(lambda: 0)
    else:
        word_freq = None
    X_data = []
    y_data = []
    label_dic = {'b': 0, 'e': 1, 'm': 2, 't': 3}
    with open(file) as fp:
        for line in fp:
            label, sentence = line.strip().split('\t')
            words = sentence.split()   
            if is_train:
                for word in words:
                    word_freq[word] += 1
            X_data.append(words)
            y_data.append(label_dic[label])
    return X_data, y_data, word_freq

def tokenize(X, word_to_id):
    """ words into ids

    :param X: words
    :return: ids
    """
    ids = [word_to_id[x] if x in word_to_id else 0 for x in X]
    return ids

if __name__ == '__main__':
    file_train = './../chapter06/train.feature.txt'
    file_dev = './../chapter06/dev.feature.txt'
    file_test = './../chapter06/test.feature.txt'

    X_train, y_train, word_freq = obtain_data(file_train, is_train=True)
    print(f'Before tokenized: {X_train[0]}')

    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    word_to_id = {}
    V = 0
    for w_f in sorted_word_freq:
        word, freq = w_f
        if freq >= 2:
            V += 1
            word_to_id[word] = V
        else:
            word_to_id[word] = 0

    pickle.dump(word_to_id, open('./word_to_id.pkl', 'wb'))

    print(f'After tokenized: {tokenize(X_train[0], word_to_id)}')

"""
Before tokenized: ['us', 'urges', 'china', 'allow', 'bigger', 'market', 'role', 'valuing', 'yuan', '(1)']
After tokenized: [3, 1662, 24, 1443, 922, 132, 192, 0, 1278, 7]
"""

