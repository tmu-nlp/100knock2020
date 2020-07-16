import pandas as pd
import numpy as np
from keras.layers import Input, Embedding, SimpleRNN, Softmax, Dense
from keras.models import Sequential
from keras.losses import CategoricalCrossentropy
from keras.preprocessing.sequence import pad_sequences
from zzz.chapter09.knock80 import text_encode

PATH = '/Users/zz_zhang/勉強会/100本ノック/100knock2020/zzz/chapter06/'
DW = 300
DH = 50
MAX_LEN = 20

if __name__ == '__main__':
    train = pd.read_csv(PATH + 'train.txt', sep='\t')
    val = pd.read_csv(PATH + 'valid.txt', sep='\t')
    test = pd.read_csv(PATH + 'test.txt', sep='\t')

    train_onehot, train_label, val_onehot, val_label, test_onehot, test_label, vocab, word_index = text_encode(train,
                                                                                                               val,
                                                                                                               test,
                                                                                                               type='onehot')

    print(train_onehot[:4])

    model = Sequential()

    model.add(Embedding(len(vocab) + 1, DW))
    model.add(SimpleRNN(DH, dropout=0.2, recurrent_dropout=0.1))
    model.add(Dense(4, activation='softmax'))

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=["accuracy"],
    )

    # model.fit(train_onehot, train_label)

    score = model.evaluate(test_onehot, test_label)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
