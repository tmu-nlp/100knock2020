"""
73. 確率的勾配降下法による学習Permalink
確率的勾配降下法（SGD: Stochastic Gradient Descent）を用いて，行列Wを学習せよ．
なお，学習は適当な基準で終了させればよい（例えば「100エポックで終了」など）
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

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
    def fit(self, tr_data, va_data, epochs):
        for i in range(epochs):
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

            self.model.eval()
            with torch.no_grad():
                sorce, target = next(iter(va_data))
                output = self.model.forward(sorce)
                valid_loss = self.criterion(output, target)

            print(f"epoch:{i+1}: training loss: {train_loss:.5f}, valid loss: {valid_loss:.5f}")
        torch.save(self.model, "sgc_model")



if __name__ == "__main__":
    train_dataset = CreateData(X_train, Y_train)
    valid_dataset = CreateData(X_valid, Y_valid)
    #for文並ぶたびに順番変わる
    train_dataset = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_dataset = DataLoader(valid_dataset, batch_size=1, shuffle=True)


    sgc = SGC()
    sgc.fit(train_dataset, valid_dataset, 10)

"""
epoch:1: training loss: 0.53334, valid loss: 0.01180
epoch:2: training loss: 0.37215, valid loss: 0.06297
epoch:3: training loss: 0.33875, valid loss: 0.19412
epoch:4: training loss: 0.31995, valid loss: 0.01432
epoch:5: training loss: 0.30759, valid loss: 0.39584
epoch:6: training loss: 0.29916, valid loss: 0.09775
epoch:7: training loss: 0.29308, valid loss: 0.10928
epoch:8: training loss: 0.28777, valid loss: 0.01213
epoch:9: training loss: 0.28384, valid loss: 1.00000
epoch:10: training loss: 0.27987, valid loss: 0.00025
"""

