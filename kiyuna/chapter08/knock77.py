"""
77. ミニバッチ化
問題76のコードを改変し，B事例ごとに損失・勾配を計算し，行列Wの値を更新せよ（ミニバッチ化）．
Bの値を1,2,4,8,…と変化させながら，1エポックの学習に要する時間を比較せよ．
"""
import logging
import os
import sys

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from knock70 import MyDataset
from knock71 import SingleLayerNet

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.time import Timer  # noqa: E402 isort:skip

logging.basicConfig(level=logging.DEBUG)


def run(train, valid, net, criterion, optimizer, batch_size=None, epochs=1):
    if not batch_size:
        batch_size = len(train)
    logging.info("[batch_size: %d]", batch_size)

    dataloader_train = DataLoader(train, batch_size=batch_size, shuffle=True)

    log = {
        "loss_train": [],
        "loss_valid": [],
        "acc_train": [],
        "acc_valid": [],
    }
    for epoch in range(1, epochs + 1):
        logging.info(f"At epoch {epoch:<3}")
        with Timer(verbose=False) as t:
            net.train()
            loss_train = 0
            acc_train = 0
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
                acc_train += (torch.argmax(logits, dim=1) == y).sum().item()
            loss_train /= i
            acc_train /= len(train)
            log["loss_train"].append(loss_train)
            log["acc_train"].append(acc_train)

            net.eval()
            with torch.no_grad():
                X, y = valid.X, valid.y
                logits = net.forward(X)
                loss_valid = criterion(logits, y)
                acc_valid = (torch.argmax(logits, dim=1) == y).sum().item() / len(valid)
                log["loss_valid"].append(loss_valid)
                log["acc_valid"].append(acc_valid)

        logging.info(f"  Time: {t.msecs:f} [msec]")
        logging.info(f"  Train -- Loss: {loss_train:f} Acc: {acc_train:f}")
        logging.info(f"  Valid -- Loss: {loss_train:f} Acc: {acc_train:f}")


if __name__ == "__main__":
    train = torch.load("./data/train.pt")
    valid = torch.load("./data/valid.pt")
    d = train.X.shape[1]
    L = len(torch.unique(train.y))
    for b in [1, 2, 4, 8, 16, 32, None]:
        net = SingleLayerNet(d, L)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=1e-1)
        run(train, valid, net, criterion, optimizer, batch_size=b)


"""result
INFO:root:[batch_size: 1]
INFO:root:At epoch 1  
INFO:root:  Time: 7842.226744 [msec]
INFO:root:  Train -- Loss: 0.520309 Acc: 0.813647
INFO:root:  Valid -- Loss: 0.520309 Acc: 0.813647
INFO:root:[batch_size: 2]
INFO:root:At epoch 1  
INFO:root:  Time: 2485.739231 [msec]
INFO:root:  Train -- Loss: 0.609950 Acc: 0.782853
INFO:root:  Valid -- Loss: 0.609950 Acc: 0.782853
INFO:root:[batch_size: 4]
INFO:root:At epoch 1  
INFO:root:  Time: 1388.149500 [msec]
INFO:root:  Train -- Loss: 0.689351 Acc: 0.754025
INFO:root:  Valid -- Loss: 0.689351 Acc: 0.754025
INFO:root:[batch_size: 8]
INFO:root:At epoch 1  
INFO:root:  Time: 755.405188 [msec]
INFO:root:  Train -- Loss: 0.843706 Acc: 0.704979
INFO:root:  Valid -- Loss: 0.843706 Acc: 0.704979
INFO:root:[batch_size: 16]
INFO:root:At epoch 1  
INFO:root:  Time: 447.675705 [msec]
INFO:root:  Train -- Loss: 1.033986 Acc: 0.618027
INFO:root:  Valid -- Loss: 1.033986 Acc: 0.618027
INFO:root:[batch_size: 32]
INFO:root:At epoch 1  
INFO:root:  Time: 296.557665 [msec]
INFO:root:  Train -- Loss: 1.159062 Acc: 0.568701
INFO:root:  Valid -- Loss: 1.159062 Acc: 0.568701
INFO:root:[batch_size: 10684]
INFO:root:At epoch 1  
INFO:root:  Time: 160.854340 [msec]
INFO:root:  Train -- Loss: 1.725370 Acc: 0.292868
INFO:root:  Valid -- Loss: 1.725370 Acc: 0.292868
"""
