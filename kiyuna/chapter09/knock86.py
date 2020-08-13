"""
86. 畳み込みニューラルネットワーク (CNN)
(ry
"""
import os
import random
import sys

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from knock80 import MyDataset, get_V

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.pickle import load  # noqa: E402 isort:skip


d_w = 300
d_h = 50
V = get_V()
L = 4


class CNN(nn.Module):
    def __init__(self, d_w, d_h, L, emb, nonlinearity="relu"):
        super().__init__()
        self.emb = nn.Parameter(emb)
        self.conv = nn.Conv1d(d_w, d_h, kernel_size=3, stride=1, padding=1)
        self.g = torch.tanh if nonlinearity == "tanh" else nn.functional.relu
        self.fc = nn.Linear(d_h, L)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        emb = (x @ self.emb.T).permute(1, 2, 0)
        p = self.g(self.conv(emb))
        c = nn.functional.max_pool1d(p, p.shape[2]).squeeze(2)
        o = self.softmax(self.fc(c))
        return o

    def show_params(self):
        print("-" * 63)
        for param_name, param in self.named_parameters():
            print(f"{param_name:23}", param.shape)
        print("-" * 63)


if __name__ == "__main__":
    train = torch.load("./data/train.pt")
    emb = torch.Tensor(d_w, V).normal_()
    cnn = CNN(d_w, d_h, L, emb)
    cnn.show_params()
    output = cnn(train.X[0].unsqueeze(1))
    print(output, output.shape, sep=", ")


"""result
---------------------------------------------------------------
emb                     torch.Size([300, 7403])
conv.weight             torch.Size([50, 300, 3])
conv.bias               torch.Size([50])
fc.weight               torch.Size([4, 50])
fc.bias                 torch.Size([4])
---------------------------------------------------------------
tensor([[0.5747, 0.1153, 0.0783, 0.2316]], grad_fn=<SoftmaxBackward>), torch.Size([1, 4])
"""
