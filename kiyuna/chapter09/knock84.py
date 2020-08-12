"""
84. 単語ベクトルの導入
事前学習済みの単語ベクトル
（例えば，Google Newsデータセット（約1,000億単語）での学習済み単語ベクトル）で
単語埋め込みemb(x)を初期化し，学習せよ．

[Usage]
python knock84.py 2>&1 | tee knock84.log 
"""
import logging
import os
import random
import sys

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from knock80 import MyDataset, _list_valid_words, get_V
from knock81 import RNN
from knock82 import run_eval, run_train

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
    rnn = RNN(d_w, d_h, L, emb)
    rnn = run_train(train, valid, rnn, epochs=11, lr=1e-1, batch_size=32, device=device)
    loss, acc = run_eval(rnn, test, device=device)
    print(f"Accuracy (test): {acc:f}, Loss (test): {loss:f}")
