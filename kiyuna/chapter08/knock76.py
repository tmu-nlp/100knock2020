"""
76. チェックポイント
問題75のコードを改変し，各エポックのパラメータ更新が完了するたびに，
チェックポイント（学習途中のパラメータ（重み行列など）の値や最適化アルゴリズムの内部状態）を
ファイルに書き出せ．
"""
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from knock70 import MyDataset
from knock71 import SingleLayerNet

os.makedirs("./checkpoints", exist_ok=True)


def run(epochs=10):
    train = torch.load("./data/train.pt")
    valid = torch.load("./data/valid.pt")

    d = train.X.shape[1]
    L = len(torch.unique(train.y))

    net = SingleLayerNet(d, L)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-1)

    dataloader_train = DataLoader(train, batch_size=len(train), shuffle=True)

    log = {
        "loss_train": [], "loss_valid": [],
        "acc_train": [], "acc_valid": [],
    }
    for epoch in range(1, epochs + 1):
        print(f"At epoch {epoch:<3}")
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

        print(f'  Train -- Loss: {loss_train:f} Acc: {acc_train:f}')
        print(f'  Valid -- Loss: {loss_train:f} Acc: {acc_train:f}')
        state_dict = {
            "epoch": epoch,
            "net_state": net.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        torch.save(state_dict, f"./checkpoints/single_{epoch}.pt")

if __name__ == '__main__':
    run()


"""result
At epoch 1  
  Train -- Loss: 2.006530 Acc: 0.218645
  Valid -- Loss: 2.006530 Acc: 0.218645
At epoch 2  
  Train -- Loss: 1.978520 Acc: 0.223418
  Valid -- Loss: 1.978520 Acc: 0.223418
At epoch 3  
  Train -- Loss: 1.952119 Acc: 0.229221
  Valid -- Loss: 1.952119 Acc: 0.229221
At epoch 4  
  Train -- Loss: 1.927229 Acc: 0.236148
  Valid -- Loss: 1.927229 Acc: 0.236148
At epoch 5  
  Train -- Loss: 1.903738 Acc: 0.241389
  Valid -- Loss: 1.903738 Acc: 0.241389
At epoch 6  
  Train -- Loss: 1.881561 Acc: 0.249906
  Valid -- Loss: 1.881561 Acc: 0.249906
At epoch 7  
  Train -- Loss: 1.860611 Acc: 0.255990
  Valid -- Loss: 1.860611 Acc: 0.255990
At epoch 8  
  Train -- Loss: 1.840799 Acc: 0.262261
  Valid -- Loss: 1.840799 Acc: 0.262261
At epoch 9  
  Train -- Loss: 1.822051 Acc: 0.267690
  Valid -- Loss: 1.822051 Acc: 0.267690
At epoch 10 
  Train -- Loss: 1.804294 Acc: 0.273587
  Valid -- Loss: 1.804294 Acc: 0.273587
"""
