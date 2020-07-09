from zzz.chapter08.knock70 import Data
from zzz.chapter08.knock71 import OneLayerNN
from gensim import models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_PATH = '/Users/zz_zhang/勉強会/100本ノック/100knock2020/zzz/chapter06/'
TRAIN_FILE = 'train.txt'
VAL_FILE = 'valid.txt'
TEST_FILE = 'test.txt'

WORD_VEC_PATH = '/Users/zz_zhang/勉強会/100本ノック/100knock2020/zzz/chapter07/'
WORD_VEC_FILE = 'GoogleNews-vectors-negative300.bin'


class MultiLayerNN(OneLayerNN):
    def __init__(self, dimensions, model_name='multi_layer.pkl', gpu=False):
        super(OneLayerNN, self).__init__()

        self.hidden_layers = []
        for index in range(1, len(dimensions) - 1):
            self.hidden_layers.append(
                nn.Linear(dimensions[index - 1], dimensions[index])
            )
        self.out = nn.Linear(dimensions[-2], dimensions[-1])

        self.history_accuracy = []
        self.history_loss = []
        self.history_accuracy_val = []
        self.history_loss_val = []

        self.best_accuracy = 0.0
        self.model_name = model_name

        self.gpu = gpu

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        output = self.out(x)
        return output


if __name__ == '__main__':
    train = Data(DATA_PATH + TRAIN_FILE)
    val = Data(DATA_PATH + VAL_FILE)
    test = Data(DATA_PATH + TEST_FILE)
    print('Data loading finished.')

    word_vec = models.KeyedVectors.load_word2vec_format(WORD_VEC_PATH + WORD_VEC_FILE, binary=True)
    print('Word vector loading finished.')

    train.embedding(word_vec)
    val.embedding(word_vec)
    test.embedding(word_vec)
    print('Text embedding finished.')

    x_train = torch.from_numpy(np.array(train.x)).float()
    y_train = torch.from_numpy(np.array(train.y)).squeeze()
    x_val = torch.from_numpy(np.array(val.x)).float()
    y_val = torch.from_numpy(np.array(val.y)).squeeze()
    model = MultiLayerNN([300, 256, 64, 4])

    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    loss_func = torch.nn.CrossEntropyLoss()
    model.fit(x_train, y_train, optimizer, loss_func, epoch=100, batch_size=-1, val_data=(x_val, y_val))

    pred_y = model.predict(x_train)
    print('accuracy(train): {}'.format(model.score_accuracy(y_train, pred_y)))

    pred_y = model.predict(x_val)
    print('accuracy(val): {}'.format(model.score_accuracy(y_val, pred_y)))

    model.plot_history()
