"""
88. パラメータチューニング
問題85や問題87のコードを改変し，ニューラルネットワークの形状やハイパーパラメータを調整しながら，
高性能なカテゴリ分類器を構築せよ．

[Usage]
python knock88.py 2>&1 | tee knock88.log

[MEMO]
うまくいってなさそう？
"""
import os
import random
import sys
from pprint import pprint

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import optuna
from knock80 import MyDataset, _list_valid_words, get_V
from knock81 import RNN
from knock82 import collate_fn, run_eval, run_train
from knock86 import CNN

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.pickle import load  # noqa: E402 isort:skip


d_w = 300
d_h = 50
V = get_V()
L = 4
device = torch.device("cpu")


def objective(trial):
    nl = trial.suggest_int("num_layers", 1, 2)
    bi = trial.suggest_categorical("bidirectional", [False, True])
    model = RNN(d_w, d_h, L, emb, num_layers=nl, bidirectional=bi)

    lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    bs = trial.suggest_categorical("batch_size", [32, -1])
    model = run_train(train, valid, model, epochs=11, lr=lr, batch_size=bs, device=device)

    loss_eval, acc_eval = run_eval(model, valid, device=device)
    return acc_eval


if __name__ == "__main__":
    emb = torch.Tensor(d_w, V).normal_()
    wv = load("chap07-embeddings")
    for i, word in enumerate(_list_valid_words()):
        if word in wv:
            wv_word = wv[word]
            wv_word.flags["WRITEABLE"] = True
            emb[:, i] = torch.from_numpy(wv_word)

    train = torch.load("./data/train.pt")
    valid = torch.load("./data/valid.pt")
    test = torch.load("./data/test.pt")

    study = optuna.create_study()
    study.optimize(objective, n_trials=30)

    print("best_value:", study.best_value)
    print("best_params:", study.best_params)
    pprint(study.best_trial)
