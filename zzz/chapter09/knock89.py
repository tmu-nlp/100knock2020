import json
import numpy as np
from random import choice
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import re, os
import pandas as pd
import codecs
from keras.layers import *
from keras.models import Model
import keras.backend as K

maxlen = 100
config_path = '/Users/zz_zhang/勉強会/100本ノック/100knock2020/zzz/chapter09/uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/Users/zz_zhang/勉強会/100本ノック/100knock2020/zzz/chapter09/uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/Users/zz_zhang/勉強会/100本ノック/100knock2020/zzz/chapter09/uncased_L-12_H-768_A-12/vocab.txt'
PATH = '/Users/zz_zhang/勉強会/100本ノック/100knock2020/zzz/chapter06/'

def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = range(len(self.data))
            # np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i:i + 1]
                text = str(d['text'])
                # print(text)
                x1, x2 = tokenizer.encode(first=text)
                y = int(d['label'])
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []

def load_data(filename):
    data = pd.DataFrame(columns=['text', 'label'])
    label_dict = {'b': 0, 't': 1, 'e': 2, 'm': 3}

    raw_data = pd.read_csv(filename, sep='\t')
    for index, item in raw_data.iterrows():
        data.append({'text': item['TITLE'],
                     'label': label_dict[item['CATEGORY']]},
                    ignore_index=True
                    )
    return data

if __name__ == '__main__':
    train_data = load_data(PATH + 'train.txt')
    val_data = load_data(PATH + 'valid.txt')
    test_data = load_data(PATH + 'test.txt')

    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    tokenizer = Tokenizer(token_dict)

    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)
    p = Dense(1, activation='sigmoid')(x)

    model = Model([x1_in, x2_in], p)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    model.summary()

    train_D = data_generator(train_data)
    val_D = data_generator(val_data)

    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=5,
        validation_data=val_D.__iter__(),
        validation_steps=len(val_D)
    )
