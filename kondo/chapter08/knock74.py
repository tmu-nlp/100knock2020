"""
74. 正解率の計測Permalink
問題73で求めた行列を用いて学習データおよび評価データの事例を分類したとき，その正解率をそれぞれ求めよ．
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

class CreateData():
    def __init__(self, x_data, y_data):
        self.x = x_data
        self.y = y_data

    def __len__(self):                      #len()でサイズを返す
        return len(self.y)

    def __getitem__(self, idx):             #getitem()で指定されたインデックスの要素を返す
        return [self.x[idx], self.y[idx]]

def calculate_accuracy(model, data):
    model.eval()
    total = 0
    cor = 0

    with torch.no_grad():
        for sorce, target in data:
            outputs = model(sorce)
            pred = torch.argmax(outputs, dim=-1)
            total += len(sorce)
            #各要素一致してるかどうかの二値判定を足し合わせてtensorの形からその数だけをとる
            cor += (pred == target).sum().item()
    return cor/total

if __name__ == "__main__":
    model = torch.load('sgc_model')

    train_dataset = CreateData(X_train, Y_train)
    test_dataset = CreateData(X_test, Y_test)
    #for文並ぶたびに順番変わる
    train_dataset = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataset = DataLoader(test_dataset, batch_size=1, shuffle=True)
    acc_train = calculate_accuracy(model, train_dataset)
    acc_test = calculate_accuracy(model, test_dataset)
    print(f"training_accuracy: {acc_train:.5f}")
    print(f"test_accuracy: {acc_test:.5f}")

"""
training_accuracy: 0.90640
test_accuracy: 0.87650
"""