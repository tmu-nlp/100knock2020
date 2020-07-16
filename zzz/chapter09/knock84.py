from gensim import models
from keras.layers import Embedding, SimpleRNN, Dense
from keras.models import Sequential
import numpy as np
import pandas as pd
from zzz.chapter09.knock80 import text_encode

PATH = '/Users/zz_zhang/勉強会/100本ノック/100knock2020/zzz/chapter06/'

WORD_VEC_PATH = '/Users/zz_zhang/勉強会/100本ノック/100knock2020/zzz/chapter07/'
WORD_VEC_FILE = 'GoogleNews-vectors-negative300.bin'

DW = 300
DH = 50


def get_embedding_matrix(word_vec, word_index):
    embedding_matrix = np.zeros((len(word_index) + 1, DW))
    for (word, i) in word_index.items():
        try:
            embedding_vector = word_vec.get_vector(word)
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        except:
            #     print('word not in word_vec')
            pass
    return embedding_matrix


if __name__ == '__main__':
    train = pd.read_csv(PATH + 'train.txt', sep='\t')
    val = pd.read_csv(PATH + 'valid.txt', sep='\t')
    test = pd.read_csv(PATH + 'test.txt', sep='\t')
    train_id, train_label, val_id, val_label, test_id, test_label, vocab, word_index = text_encode(train, val, test,
                                                                                                   type='seq')
    word_vec = models.KeyedVectors.load_word2vec_format(WORD_VEC_PATH + WORD_VEC_FILE, binary=True)

    embedding_matrix = get_embedding_matrix(word_vec, word_index)
    print(embedding_matrix.shape)

    model = Sequential()

    model.add(Embedding(len(vocab) + 1, DW, weights=[embedding_matrix], trainable=False))
    model.add(SimpleRNN(DH, dropout=0.2, recurrent_dropout=0.1))
    model.add(Dense(4, activation='softmax'))

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=["accuracy"],
    )

    model.fit(train_id, train_label, validation_data=(val_id, val_label))
