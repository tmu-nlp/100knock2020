"""
80. ID番号への変換
問題51で構築した学習データ中の単語にユニークなID番号を付与したい．
学習データ中で最も頻出する単語に1，2番目に頻出する単語に2，……といった方法で，
学習データ中で2回以上出現する単語にID番号を付与せよ．
そして，与えられた単語列に対して，ID番号の列を返す関数を実装せよ．
ただし，出現頻度が2回未満の単語のID番号はすべて0とせよ．
"""
import os
import random
import sys
from collections import Counter

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.pickle import load  # noqa: E402 isort:skip


seed = 2020
random.seed(seed)
np.random.seed(seed)  
torch.manual_seed(seed)  

inv_table = None
V = -1


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, V):
        self.X = X
        self.y = torch.tensor(y)
        self.V = X[0].shape[1]
        # self.idxs = torch.arange(len(self))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # idx = self.idxs[idx]
        y = torch.nn.functional.one_hot(self.y[idx], num_classes=self.V)
        return torch.cat((y.reshape(1, -1).float(), self.X[idx]), 0)

    def split_yX(self, yX):
        return torch.argmax(yX[0], dim=1), yX[1:]
  

    # def shuffle(self):
    #     self.idxs = self.idxs[torch.randperm(len(self))]


def _read_file(split_name):
    with open(f"../chapter06/{split_name}.feature.txt") as f:
        for line in f:
            label, sent = line.strip().split('\t')
            yield label, sent


def _list_valid_words():
    cnter = Counter()
    for _, sent in _read_file('train'):
        words = sent.split(' ')
        cnter += Counter(words)
    del cnter['']
    for word in tuple(cnter.keys()):
        if cnter[word] == 1:
            del cnter[word]
    assert '' not in cnter and min(cnter.values()) > 1
    valid_words, cnts = zip(*cnter.most_common())
    return valid_words


def _build_inv_table():
    global inv_table, V
    valid_words = _list_valid_words()
    inv_table = {word: i for i, word in enumerate(valid_words, start=1)}
    V = len(valid_words) + 1    # add UNK


def word2id(word):
    if inv_table is None:
        _build_inv_table()
    return inv_table.get(word, 0)


def sent2id(words):
    return [word2id(word) for word in words]


def get_V():
    if inv_table is None:
        _build_inv_table()
    return V


def one_hot_encode(words):
    ids = sent2id(words)
    tensor = torch.nn.functional.one_hot(torch.tensor(ids), num_classes=get_V())
    return tensor.float()


def build_and_save(split_name, debug=False):
    os.makedirs("./data", exist_ok=True)
    if not debug and os.path.exists(f"./data/{split_name}.pt"):
        return
    X, y = [], []
    for label, sent in _read_file('train'):
        words = sent.split(' ')
        X += [one_hot_encode(words)]
        y += ["btem".index(label)]
    data = MyDataset(X, y, get_V())
    torch.save(data, f"./data/{split_name}.pt")
    return data


if __name__ == "__main__":
    one_hot = one_hot_encode(["...", "", "acl", "CS", "exe"])
    print(one_hot, one_hot.shape, type(one_hot), sep=', ')

    split_names = ["train", "valid", "test"]
    for split_name in split_names:
        build_and_save(split_name, debug=True)

    valid = torch.load('./data/valid.pt')
    print(valid[0], valid[0].shape, sep=', ')
    # valid.shuffle()
    # print(valid[0])


"""result
tensor([[0., 1., 0.,  ..., 0., 0., 0.],
        [1., 0., 0.,  ..., 0., 0., 0.],
        [1., 0., 0.,  ..., 0., 0., 0.],
        [1., 0., 0.,  ..., 0., 0., 0.],
        [1., 0., 0.,  ..., 0., 0., 0.]]), torch.Size([5, 7403]), <class 'torch.Tensor'>
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [1., 0., 0.,  ..., 0., 0., 0.],
        [0., 1., 0.,  ..., 0., 0., 0.]]), torch.Size([13, 7403])
"""
