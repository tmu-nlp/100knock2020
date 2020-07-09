from zzz.chapter08.knock70 import Data
import torch.utils.data as TorchData
from gensim import models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

DATA_PATH = '/Users/zz_zhang/勉強会/100本ノック/100knock2020/zzz/chapter06/'
TRAIN_FILE = 'train.txt'
VAL_FILE = 'valid.txt'
TEST_FILE = 'test.txt'

WORD_VEC_PATH = '/Users/zz_zhang/勉強会/100本ノック/100knock2020/zzz/chapter07/'
WORD_VEC_FILE = 'GoogleNews-vectors-negative300.bin'


class OneLayerNN(nn.Module):
    def __init__(self, d_in, d_out, model_name='one_layer.pkl', gpu=False):
        super(OneLayerNN, self).__init__()
        self.out = nn.Linear(d_in, d_out)

        self.history_accuracy = []
        self.history_loss = []
        self.history_accuracy_val = []
        self.history_loss_val = []

        self.best_accuracy = 0.0
        self.model_name = model_name

        self.gpu = gpu

    def forward(self, x):
        output = self.out(x)
        return output

    def fit(self, x, y, optimizer, loss_func, epoch=1, batch_size=-1, val_data=None, check_point=False):
        if val_data is not None:
            x_val = val_data[0]
            y_val = val_data[1]

        if batch_size == -1:
            if self.gpu:
                x = x.cuda()
                y = y.cuda()
            for _ in range(epoch):
                out = self(x)
                loss = loss_func(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                out = self(x)
                loss_value = loss_func(out, y).detach().numpy().squeeze()
                accuracy = self.score_accuracy(y, self.predict(x))
                self.history_loss.append(loss_value)
                self.history_accuracy.append(accuracy)
                if val_data is not None:
                    out = self(x_val)
                    loss_value = loss_func(out, y_val).detach().numpy().squeeze()
                    accuracy = self.score_accuracy(y_val, self.predict(x_val))
                    self.history_loss_val.append(loss_value)
                    self.history_accuracy_val.append(accuracy)

                    if check_point:
                        if accuracy > self.best_accuracy:
                            self.best_accuracy = accuracy
                            self.save(self.model_name)
                else:
                    if check_point:
                        if accuracy > self.best_accuracy:
                            self.best_accuracy = accuracy
                            self.save(self.model_name)

        else:
            torch_dataset = TorchData.TensorDataset(x, y)
            loader = TorchData.DataLoader(
                dataset=torch_dataset,
                batch_size=batch_size,
                shuffle=True,
            )
            for _ in range(epoch):
                for step, (batch_x, batch_y) in enumerate(loader):
                    if self.gpu:
                        batch_x = batch_x.cuda()
                        batch_y = batch_y.cuda()

                    out = model(batch_x)
                    loss = loss_func(out, batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                out = self(x)
                loss_value = loss_func(out, y).detach().numpy().squeeze()
                accuracy = self.score_accuracy(y, self.predict(x))
                self.history_loss.append(loss_value)
                self.history_accuracy.append(accuracy)
                if val_data is not None:
                    out = self(x_val)
                    loss_value = loss_func(out, y_val).detach().numpy().squeeze()
                    accuracy = self.score_accuracy(y_val, self.predict(x_val))
                    self.history_loss_val.append(loss_value)
                    self.history_accuracy_val.append(accuracy)

                    if check_point:
                        if accuracy > self.best_accuracy:
                            self.best_accuracy = accuracy
                            self.save(self.model_name)
                else:
                    if check_point:
                        if accuracy > self.best_accuracy:
                            self.best_accuracy = accuracy
                            self.save(self.model_name)

    def predict(self, x):
        if self.gpu:
            x = x.cuda()
            out = self(x)
            prediction = torch.max(F.softmax(out), 1)[1].cuda()
            pred_y = prediction.data.numpy().squeeze()
        else:
            out = self(x)
            prediction = torch.max(F.softmax(out), 1)[1]
            pred_y = prediction.data.numpy().squeeze()
        return pred_y

    def score_accuracy(self, y, pred_y):
        total = len(y)
        temp_y = y.data.numpy().squeeze()
        different = temp_y - pred_y
        correct = np.count_nonzero(different == 0)
        return float(correct) / total

    def plot_history(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x_axis = [i + 1 for i in range(len(self.history_accuracy))]
        if len(self.history_accuracy) > 0:
            ax.plot(x_axis, self.history_accuracy, label='accuracy_train')
        if len(self.history_loss) > 0:
            ax.plot(x_axis, self.history_loss, label='loss_train')
        if len(self.history_accuracy_val) > 0:
            ax.plot(x_axis, self.history_accuracy_val, label='accuracy_val')
        if len(self.history_loss_val) > 0:
            ax.plot(x_axis, self.history_loss_val, label='loss_val')
        ax.legend(loc='upper right')
        plt.show()

    def save(self, filename):
        torch.save(self, filename)


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
    y_train = torch.from_numpy(np.array(train.y))

    model = OneLayerNN(300, 4)

    # pred_y = model.predict(x_train)
    # print('accuracy: {}'.format(model.score_accuracy(y_train, pred_y)))

    print(F.softmax(model(x_train)))