"""
78. GPU上での学習Permalink
問題77のコードを改変し，GPU上で学習を実行せよ．
"""

import time
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from matplotlib import pyplot as plt

X_train_data = "X_train.npy"
Y_train_data = "Y_train.npy"
X_valid_data = "X_valid.npy"
Y_valid_data = "Y_valid.npy"
X_test_data = "X_test.npy"
Y_test_data = "Y_test.npy"

X_train = np.load(file=X_train_data)
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = np.load(file=Y_train_data)
Y_train = torch.tensor(Y_train, dtype=torch.int64)
X_valid = np.load(file=X_valid_data)
X_valid = torch.tensor(X_valid, dtype=torch.float32)
Y_valid = np.load(file=Y_valid_data)
Y_valid = torch.tensor(Y_valid, dtype=torch.int64)
X_test = np.load(file=X_test_data)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = np.load(file=Y_test_data)
Y_test = torch.tensor(Y_test, dtype=torch.int64)

d = 300
L = 4

class SLPNet(torch.nn.Module):
    def __init__(self, input_size, output_size):
        torch.manual_seed(7)
        #nn.Moduleのinitを引っ張ってくる
        super().__init__()
        self.fc = torch.nn.Linear(input_size, output_size, bias=False)
        #重みを平均0分散1で初期化
        torch.nn.init.normal_(self.fc.weight)

    def forward(self, x):
        x = self.fc(x)
        return x

#データセットをlistみたいな形にしておけるクラス
class CreateData():
    def __init__(self, x_data, y_data):
        self.x = x_data
        self.y = y_data

    def __len__(self):                      #len()でサイズを返す
        return len(self.y)

    def __getitem__(self, idx):             #getitem()で指定されたインデックスの要素を返す
        return [self.x[idx], self.y[idx]]

class SGC():
    def __init__(self):
        self.model = SLPNet(d, L)
        self.criterion = torch.nn.CrossEntropyLoss()
        #オプティマイザ
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
    def fit(self, tr_data, va_data, epochs, device=None):
        self.model.to(device)
        train_log = []
        valid_log = []
        total_time = 0
        for i in range(epochs):

            s_time = time.time()

            self.model.train()
            total_loss=0.0
            for j, (sorce, target) in enumerate(tr_data):
                self.optimizer.zero_grad()          #勾配の初期化

                output = self.model.forward(sorce)  #順伝播
                loss = self.criterion(output, target)
                loss.backward()                     #逆伝播
                self.optimizer.step()               #重み更新

                total_loss += loss.item()
            train_loss = total_loss/j              #バッチ単位のロス

            train_loss, train_acc = self.calculate_loss_acc(tr_data, device)
            valid_loss, valid_acc = self.calculate_loss_acc(va_data, device)
            train_log.append([train_loss, train_acc])
            valid_log.append([valid_loss, valid_acc])

            torch.save({"epoch": i, "model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict()}, f'checkpoint{i+1}.pt')

            e_time = time.time()

            print(f"epoch: {i+1}, train loss: {train_loss:.5f}, train accuracy: {train_acc}, valid loss: {valid_loss}, valid accuracy: {valid_acc}, time: {(e_time-s_time):.5f}sec")

            total_time += e_time-s_time
        print(f"1エポックあたり{total_time/epochs}s\n")

    def calculate_loss_acc(self, data, device):
        self.model.eval()
        loss = 0
        total = 0
        cor = 0
        with torch.no_grad():
            for sorce, target in data:
                output = self.model(sorce)
                loss += self.criterion(output, target).item()
                pred = torch.argmax(output, dim=-1)
                total += len(sorce)
                cor += (pred == target).sum().item()
        return loss/len(data), cor/total

    def plot(self, train_log, valid_log, flag):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].plot(np.array(train_log).T[0], label="train")
        ax[0].plot(np.array(valid_log).T[0], label="valid")
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("loss")
        ax[0].legend()
        ax[1].plot(np.array(train_log).T[1], label="train")
        ax[1].plot(np.array(valid_log).T[1], label="valid")
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("accuracy")
        ax[1].legend()
        if flag == 1:
            plt.show()
        else:
            plt.pause(.01)

if __name__ == "__main__":
    train_dataset = CreateData(X_train, Y_train)
    valid_dataset = CreateData(X_valid, Y_valid)
    test_dataset = CreateData(X_test, Y_test)

    #GPUの指定
    device = torch.device('cuda')

    lis = [1, 2, 4, 8]
    sgc = [SGC(), SGC(), SGC(), SGC()]
    for i, batch_size in enumerate(lis):
        print(f"batch size: {batch_size}")
        train_l_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_l_dataset = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
        sgc[i].fit(train_l_dataset, valid_l_dataset, 10, device=device)
