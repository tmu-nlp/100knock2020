"""
87. 確率的勾配降下法によるCNNの学習Permalink
確率的勾配降下法（SGD: Stochastic Gradient Descent）を用いて，
問題86で構築したモデルを学習せよ．
訓練データ上の損失と正解率，評価データ上の損失と正解率を表示しながらモデルを学習し，
適当な基準（例えば10エポックなど）で終了させよ．

[Usage]
python knock87.py 2>&1 | tee knock87.log
"""
import logging
import os
import random
import sys

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from knock80 import MyDataset, _list_valid_words, get_V
from knock82 import collate_fn, run_eval, run_train
from knock86 import CNN

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.pickle import load  # noqa: E402 isort:skip

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
    wv = load("chap07-embeddings")
    for i, word in enumerate(_list_valid_words()):
        if word in wv:
            wv_word = wv[word]
            wv_word.flags["WRITEABLE"] = True
            emb[:, i] = torch.from_numpy(wv_word)
    cnn = CNN(d_w, d_h, L, emb)
    cnn = run_train(train, valid, cnn, epochs=11, lr=1e-1, batch_size=32, device=device)
    loss, acc = run_eval(cnn, test, device=device)
    print(f"Accuracy (test): {acc:f}, Loss (test): {loss:f}")
