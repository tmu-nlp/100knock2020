"""
82. 確率的勾配降下法による学習
確率的勾配降下法（SGD: Stochastic Gradient Descent）を用いて，
問題81で構築したモデルを学習せよ．
訓練データ上の損失と正解率，評価データ上の損失と正解率を表示しながらモデルを学習し，
適当な基準（例えば10エポックなど）で終了させよ．

[Usage]
python knock82.py 2>&1 | tee knock82.log 
"""
import logging
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from knock80 import MyDataset, get_V
from knock81 import RNN

logging.basicConfig(level=logging.DEBUG)


d_w = 300
d_h = 50
V = get_V()
L = 4


def collate_fn(batch):
    x = nn.utils.rnn.pad_sequence(batch)
    return x


def run_eval(model, dataset, *, device="cpu"):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn)
    yX = next(iter(loader))
    y, X = dataset.split_yX(yX)
    y = y.to(device)
    X = X.to(device)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        output = model(X)
    loss = criterion(output, y)
    preds = torch.argmax(output, dim=1)
    acc = torch.mean((preds == y).float())
    return loss, acc


def run_train(train, valid, model, *, epochs, lr, batch_size=None, device="cpu"):
    train = torch.load("./data/train.pt")
    valid = torch.load("./data/valid.pt")
    if batch_size in [None, -1]:
        batch_size = len(train)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    log = {
        "loss_train": [], "loss_valid": [],
        "acc_train": [], "acc_valid": [],
    }
    for epoch in range(1, epochs + 1):
        logging.info(f"At epoch {epoch:<3}")
        model.train()
        loss_train = 0
        acc_train = 0
        for i, yX in enumerate(train_loader, start=1):
            # print(yX.shape)
            # => torch.Size([17, 10686, 7403])
            y, X = train.split_yX(yX)
            y = y.to(device)
            X = X.to(device)
            # 勾配の初期化
            optimizer.zero_grad()
            # 順伝播
            output = model(X)
            # ロスの計算
            loss = criterion(output, y)
            # 勾配の計算
            loss.backward()
            # パラメータの更新
            optimizer.step()
            loss_train += loss.item()
            acc_train += (torch.argmax(output, dim=1) == y).sum().item()
        loss_train /= i
        acc_train /= len(train)
        log["loss_train"].append(loss_train)
        log["acc_train"].append(acc_train)

        loss_valid, acc_valid = run_eval(model, valid, device=device)
        log["loss_valid"].append(loss_valid)
        log["acc_valid"].append(acc_valid)

        logging.info(f'  Loss (train): {loss_train:f} -- Acc (train): {acc_train:f}')
        logging.info(f'  Loss (valid): {loss_valid:f} -- Acc (valid): {acc_valid:f}')
    return model


if __name__ == "__main__":
    train = torch.load("./data/train.pt")
    valid = torch.load("./data/valid.pt")
    test = torch.load("./data/test.pt")
    emb = torch.Tensor(d_w, V).normal_()
    rnn = RNN(d_w, d_h, L, emb)
    rnn = run_train(train, valid, rnn, epochs=11, lr=1e-1)
    loss, acc = run_eval(rnn, test)
    print(f'Accuracy (test): {acc:f}, Loss (test): {loss:f}')
