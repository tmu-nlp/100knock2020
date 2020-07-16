"""
75. 損失と正解率のプロット
問題73のコードを改変し，各エポックのパラメータ更新が完了するたびに，
訓練データでの損失，正解率，検証データでの損失，正解率をグラフにプロットし，
学習の進捗状況を確認できるようにせよ．
"""
import matplotlib.pyplot as plt
import torch
import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader

from knock70 import MyDataset
from knock71 import SingleLayerNet


def run(epochs, lr):
    train = torch.load("./data/train.pt")
    valid = torch.load("./data/valid.pt")

    d = train.X.shape[1]
    L = len(torch.unique(train.y))

    net = SingleLayerNet(d, L)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)

    dataloader_train = DataLoader(train, batch_size=len(train), shuffle=True)

    log = {
        "loss_train": [],
        "loss_valid": [],
        "acc_train": [],
        "acc_valid": [],
    }
    for epoch in tqdm.tqdm(range(1, epochs + 1)):
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

    # plot
    fix, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(log["loss_train"], label="Loss Train", color="r")
    ax1.plot(log["loss_valid"], label="Loss Valid", color="m")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.plot(log["acc_train"], label="Acc Train", color="b")
    ax2.plot(log["acc_valid"], label="Acc Valid", color="c")
    ax2.set_ylabel("Accuracy")
    handler1, label1 = ax1.get_legend_handles_labels()
    handler2, label2 = ax2.get_legend_handles_labels()
    ax1.legend(handler1 + handler2, label1 + label2, loc=2, borderaxespad=0.0)
    plt.savefig(f"knock75_lr{lr:.0e}.png")


if __name__ == "__main__":
    lr = eval(input("lr -> "))
    run(epochs=30000, lr=lr)
