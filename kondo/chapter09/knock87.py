"""
87. 確率的勾配降下法によるCNNの学習Permalink
確率的勾配降下法（SGD: Stochastic Gradient Descent）を用いて，問題86で構築したモデルを学習せよ．
訓練データ上の損失と正解率，評価データ上の損失と正解率を表示しながらモデルを学習し，適当な基準（例えば10エポックなど）で終了させよ．
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from matplotlib import pyplot as plt
import time
from tqdm import tqdm

from knock80 import translate_to_id
from knock81 import get_X_Y, CreateData
from knock83 import Padsequence, train_model, plot, result
from knock85 import word2vec
from knock86 import CNN

train_path = "../chapter06/train.csv"
valid_path = "../chapter06/valid.csv"
test_path = "../chapter06/test.csv"

_, V = translate_to_id("")

VOCAB_SIZE = V+1
EMB_SIZE = 300
PADDING_IDX = V
OUTPUT_SIZE = 4
#入力データを幾つの視点からみるか(今回はword2vecによる表現のみ考えるので1個)
IN_CHANNELS = 1
OUT_CHANNELS = 100
#フィルタのサイズがこれ*emb_size
KERNEL_HEIGHTS = 3
#フィルタの動かし幅
STRIDE = 1
#どれだげパディングするか
PADDING = 1
LEARNIG_RATE = 0.005
BATCH_SIZE = 64
NUM_EPOCHS = 10

with open("id_file", encoding="utf-8") as f:
    ids = {}
    for line in f:
        id, word = line.split()
        ids[word] = int(id)

if __name__ == "__main__":
    train_X, train_Y = get_X_Y(train_path)
    valid_X, valid_Y = get_X_Y(valid_path)
    test_X, test_Y = get_X_Y(test_path)
    train_dataset = CreateData(train_X, train_Y, translate_to_id)
    valid_dataset = CreateData(valid_X, valid_Y, translate_to_id)
    test_dataset = CreateData(test_X, test_Y, translate_to_id)

    W = word2vec(ids)
    device = torch.device("cuda")

    model = CNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, IN_CHANNELS, OUT_CHANNELS, KERNEL_HEIGHTS, STRIDE, PADDING, emb_weights=W)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNIG_RATE)

    log = train_model(train_dataset, valid_dataset, model, criterion, optimizer, NUM_EPOCHS, BATCH_SIZE, collate_fn=Padsequence(PADDING_IDX), device=device)
    plot(log)
    result(train_dataset, test_dataset, model, criterion, device)