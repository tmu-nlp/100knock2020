import pandas as pd
import numpy as np
from gensim import models
from nltk.stem import PorterStemmer
import torch

DATA_PATH = '/Users/zz_zhang/勉強会/100本ノック/100knock2020/zzz/chapter06/'
TRAIN_FILE = 'train.txt'
VAL_FILE = 'valid.txt'
TEST_FILE = 'test.txt'

WORD_VEC_PATH = '/Users/zz_zhang/勉強会/100本ノック/100knock2020/zzz/chapter07/'
WORD_VEC_FILE = 'GoogleNews-vectors-negative300.bin'

class Data(object):
    def __init__(self, filename):
        data = pd.read_csv(filename, sep='\t')
        self.text = data['TITLE']
        self.label = data['CATEGORY']
        self.x = []
        self.y = []

    def embedding(self, word_vec):
        stemmer = PorterStemmer()
        vec_shape = word_vec['cat'].shape

        for (index, text) in enumerate(self.text):
            temp_vec = np.zeros(vec_shape)
            num_words = 0
            for word in text.split(' '):
                num_words += 1
                word_stem = stemmer.stem(word)
                if word_stem in word_vec:
                    temp_vec += word_vec[word_stem]

                else:
                    temp_vec += ((np.random.random(vec_shape) - 0.5) / 10)

            self.x.append(temp_vec / num_words)

        label_dict = {'b': 0, 't': 1, 'e': 2, 'm': 3}
        # label_dict = {'b': [0, 0, 0, 1],
        #               't': [0, 0, 1, 0],
        #               'e': [0, 1, 0, 0],
        #               'm': [1, 0, 0, 0]}
        for label in self.label:
            self.y.append(label_dict[label])

        self.x = pd.DataFrame(self.x)
        self.y = pd.DataFrame(self.y)
        print('Embedding finished.')


if __name__ == '__main__':
    train = Data(DATA_PATH + TRAIN_FILE)
    # val = Data(DATA_PATH + VAL_FILE)
    # test = Data(DATA_PATH + TEST_FILE)

    word_vec = models.KeyedVectors.load_word2vec_format(WORD_VEC_PATH + WORD_VEC_FILE, binary=True)

    train.embedding(word_vec)
    # val.embedding(word_vec)
    # test.embedding(word_vec)

    print(train.x.head(), train.y.head())

