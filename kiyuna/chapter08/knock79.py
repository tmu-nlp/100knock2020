"""
79. 多層ニューラルネットワーク
問題78のコードを改変し，バイアス項の導入や多層化など，ニューラルネットワークの形状を
変更しながら，高性能なカテゴリ分類器を構築せよ．
"""
import logging
import os
import sys

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from knock70 import MyDataset
from knock74 import calc_accuracy
from knock78 import run

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.time import Timer  # noqa: E402 isort:skip

logging.basicConfig(level=logging.DEBUG)


class MultiLayerNet(nn.Module):
    def __init__(self, input_size, output_size, *, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.layers = [self.fc1, self.fc2]
        self.out = nn.Linear(hidden_size, output_size)
        for layer in self.layers:
            nn.init.normal_(layer.weight, 0.0, 1.0)
        nn.init.normal_(self.out.weight, 0.0, 1.0)

    def forward(self, x):
        for layer in self.layers:
            x = nn.functional.relu(layer(x))
        return self.out(x)


if __name__ == "__main__":
    train = torch.load("./data/train.pt")
    valid = torch.load("./data/valid.pt")
    test = torch.load("./data/test.pt")
    d = train.X.shape[1]
    L = len(torch.unique(train.y))
    device = torch.device("cuda:3")
    for h in range(15):
        net = MultiLayerNet(d, L, hidden_size=2 ** h).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=1e-1)
        run(train, valid, net, criterion, optimizer, device=device, epochs=10)
        acc_test = calc_accuracy(net, test, device=device)
        print(f"Accuracy (test) : {acc_test:f} h={2**h}")


"""result
epoch=10, lr=1e-1
[knock79]
Accuracy (test) : 0.405243 h=1
Accuracy (test) : 0.403745 h=2
Accuracy (test) : 0.347566 h=4
Accuracy (test) : 0.384270 h=8
Accuracy (test) : 0.438202 h=16
Accuracy (test) : 0.501873 h=32
Accuracy (test) : 0.635206 h=64
Accuracy (test) : 0.678652 h=128
Accuracy (test) : 0.712360 h=256
Accuracy (test) : 0.744569 h=512
Accuracy (test) : 0.408989 h=1024
Accuracy (test) : 0.405243 h=2048
Accuracy (test) : 0.405243 h=4096
Accuracy (test) : 0.405243 h=8192
Accuracy (test) : 0.405243 h=16384
[konck74]
Accuracy (test) : 0.284644
"""
