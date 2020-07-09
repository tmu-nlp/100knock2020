"""
73. 確率的勾配降下法による学習
確率的勾配降下法（SGD: Stochastic Gradient Descent）を用いて，行列Wを学習せよ．
なお，学習は適当な基準で終了させればよい（例えば「100エポックで終了」など）．
"""
import logging
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from knock70 import MyDataset
from knock71 import SingleLayerNet

logging.basicConfig(level=logging.DEBUG)

os.makedirs("./model", exist_ok=True)


def run(epochs=10):
    train = torch.load("./data/train.pt")
    valid = torch.load("./data/valid.pt")

    d = train.X.shape[1]
    L = len(torch.unique(train.y))

    net = SingleLayerNet(d, L)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-1)

    dataloader_train = DataLoader(train, batch_size=len(train), shuffle=True)

    log = {"loss_train": [], "loss_valid": []}
    for epoch in range(1, epochs + 1):
        net.train()
        loss_train = 0
        for i, (X, y) in enumerate(dataloader_train, start=1):
            # 順伝播
            logits = net.forward(X)
            # ロスの計算
            loss = criterion(logits, y)
            # 勾配の初期化
            optimizer.zero_grad()
            # 勾配の計算
            loss.backward()
            # パラメータの更新
            optimizer.step()
            loss_train += loss.item()
        loss_train /= i
        log["loss_train"].append(loss_train)

        net.eval()
        with torch.no_grad():
            X, y = valid.X, valid.y
            logits = net.forward(X)
            loss_valid = criterion(logits, y)
            log["loss_valid"].append(loss_valid)

        logging.info(
            f"[epoch: {epoch:<3}] loss_train: {loss_train:f}, loss_valid: {loss_valid:f}"
        )

    torch.save(net, "./model/knock73-net.pt")


if __name__ == "__main__":
    run()


"""result
INFO:root:[epoch: 1  ] loss_train: 2.006530, loss_valid: 1.931139
INFO:root:[epoch: 2  ] loss_train: 1.978520, loss_valid: 1.904820
INFO:root:[epoch: 3  ] loss_train: 1.952119, loss_valid: 1.880004
INFO:root:[epoch: 4  ] loss_train: 1.927229, loss_valid: 1.856585
INFO:root:[epoch: 5  ] loss_train: 1.903738, loss_valid: 1.834472
INFO:root:[epoch: 6  ] loss_train: 1.881561, loss_valid: 1.813576
INFO:root:[epoch: 7  ] loss_train: 1.860611, loss_valid: 1.793809
INFO:root:[epoch: 8  ] loss_train: 1.840799, loss_valid: 1.775100
INFO:root:[epoch: 9  ] loss_train: 1.822051, loss_valid: 1.757370
INFO:root:[epoch: 10 ] loss_train: 1.804294, loss_valid: 1.740554
"""
