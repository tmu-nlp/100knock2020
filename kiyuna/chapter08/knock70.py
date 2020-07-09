"""
70. 単語ベクトルの和による特徴量
(ry
"""
import os
import random
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.pickle import load  # noqa: E402 isort:skip


seed = 2020
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X.float()
        self.y = y.reshape(-1).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_and_save(split_name, emb):
    path = f"../chapter06/{split_name}.txt"
    X, y = [], []
    for line in open(path):
        label, sent = line.strip().split("\t")
        tokens = [token for token in sent.split() if token in emb.vocab]
        if not tokens:
            continue
        X += [np.mean(emb[tokens], axis=0)]
        y += ["btem".index(label)]
    X = np.array(X)
    y = np.array(y)
    data = MyDataset(torch.from_numpy(X), torch.from_numpy(y))
    torch.save(data, f"./data/{split_name}.pt")


if __name__ == "__main__":
    emb = load("chap07-embeddings")
    split_names = ["train", "valid", "test"]
    for split_name in split_names:
        build_and_save(split_name, emb)
