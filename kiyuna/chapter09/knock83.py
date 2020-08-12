"""
83. ミニバッチ化・GPU上での学習
問題82のコードを改変し，
B事例ごとに損失・勾配を計算して学習を行えるようにせよ（Bの値は適当に選べ）．
また，GPU上で学習を実行せよ．

[Ref]
- より少ないパディングでミニバッチ学習する方法
    - https://tma15.github.io/blog/2020/03/10/pytorch%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86-%E3%82%88%E3%82%8A%E5%B0%91%E3%81%AA%E3%81%84%E3%83%91%E3%83%87%E3%82%A3%E3%83%B3%E3%82%B0%E3%81%A7%E3%83%9F%E3%83%8B%E3%83%90%E3%83%83%E3%83%81%E5%AD%A6%E7%BF%92%E3%81%99%E3%82%8B%E6%96%B9%E6%B3%95/

[Usage]
python knock83.py 2>&1 | tee knock83.log 
"""
import logging
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from knock80 import MyDataset, get_V
from knock81 import RNN
from knock82 import collate_fn, run_eval, run_train

logging.basicConfig(level=logging.DEBUG)


d_w = 300
d_h = 50
V = get_V()
L = 4


if __name__ == "__main__":
    train = torch.load("./data/train.pt")
    valid = torch.load("./data/valid.pt")
    test = torch.load("./data/test.pt")
    device = torch.device("cuda:7")
    emb = torch.Tensor(d_w, V).normal_()
    rnn = RNN(d_w, d_h, L, emb)
    rnn = run_train(train, valid, rnn, epochs=11, lr=1e-1, batch_size=32, device=device)
    loss, acc = run_eval(rnn, test, device=device)
    print(f"Accuracy (test): {acc:f}, Loss (test): {loss:f}")
