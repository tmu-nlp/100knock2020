"""
88. パラメータチューニングPermalink
問題85や問題87のコードを改変し，ニューラルネットワークの形状やハイパーパラメータを調整しながら，高性能なカテゴリ分類器を構築せよ．
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
from knock83 import Padsequence
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
OUT_CHANNELS = [50, 100, 200]
#フィルタのサイズがこれ*emb_size
KERNEL_HEIGHTS = 3
#フィルタの動かし幅
STRIDE = 1
#どれだげパディングするか
PADDING = 1
LEARNIG_RATE = [0.1, 0.01, 0.001]
BATCH_SIZE = 64
NUM_EPOCHS = 10

best_valid = 0

with open("id_file", encoding="utf-8") as f:
    ids = {}
    for line in f:
        id, word = line.split()
        ids[word] = int(id)

def train_model(tr_data, va_data, model, criterion, optimizer, epochs=10, batch_size=1, collate_fn=None, device=None):

    model.to(device)

    #collate_fnはミニバッチ化する時にサイズを揃える時に使える。基本はtensorを返すだけ
    tr_l_data = DataLoader(tr_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    va_l_data = DataLoader(va_data, batch_size=1, shuffle=False)

    #スケジューラの設定cosine関数に従って学習率をepoch数で初期値からeta_minまで小さくする
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-5, last_epoch=-1)

    total_time = 0
    train_log = []
    valid_log = []
    for i in tqdm(range(epochs)):
        s_time = time.time()

        model.train()
        total_loss=0.0

        for j, X in enumerate(tr_l_data):
            optimizer.zero_grad()          #勾配の初期化

            #デバイスの指定
            inputs = X["id_text"].to(device)
            labels = X["labels"].to(device)

            outputs = model.forward(inputs)  #順伝播
            loss = criterion(outputs, labels)
            loss.backward()                     #逆伝播
            optimizer.step()               #重み更新

            total_loss += loss.item()
        train_loss = total_loss/(j+1)              #バッチ単位のロス

        model.eval()

        train_loss, train_acc = calculate_loss_acc(tr_data, model, criterion, device)
        valid_loss, valid_acc = calculate_loss_acc(va_data, model, criterion, device)
        train_log.append([train_loss, train_acc])
        valid_log.append([valid_loss, valid_acc])

        torch.save({"epoch": i+1, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, f"checkpoinit{i+1}.pt")

        e_time = time.time()

        print(f"epoch: {i+1}, train loss: {train_loss:.5f}, train accuracy: {train_acc}, valid loss: {valid_loss}, valid accuracy: {valid_acc}, time: {(e_time-s_time):.5f}sec")

        total_time += e_time-s_time
        print(f"1エポックあたり{total_time/(i+1)}s\n")

        #検証データのロスが3エポック連続で低下しなかったら学習終了
        if i > 2 and valid_log[i-3][0] <= valid_log[i-2][0] <= valid_log[i-1][0] <= valid_log[i][0]:
            break

        scheduler.step()

    return valid_acc

def calculate_loss_acc(data, model, criterion, device=None):
    l_data = DataLoader(data, batch_size=1, shuffle=False)
    loss = 0
    total = 0
    cor = 0
    with torch.no_grad():
        for X in l_data:
            #デバイスの指定
            inputs = X["id_text"].to(device)
            labels = X["labels"].to(device)

            outputs = model(inputs)

            loss += criterion(outputs, labels)

            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            cor += (pred == labels).sum().item()
    return loss/len(l_data), cor/total


def result(train_dataset, test_dataset, model, criterion, device):
    _, train_acc = calculate_loss_acc(train_dataset, model, criterion, device)
    _, valid_acc = calculate_loss_acc(valid_dataset, model, criterion, device)
    _, test_acc = calculate_loss_acc(test_dataset, model, criterion, device)
    print(f"train accuracy: {train_acc}")
    print(f"valid accuracy: {valid_acc}")
    return train_acc, valid_acc, test_acc

if __name__ == "__main__":
    train_X, train_Y = get_X_Y(train_path)
    valid_X, valid_Y = get_X_Y(valid_path)
    test_X, test_Y = get_X_Y(test_path)
    train_dataset = CreateData(train_X, train_Y, translate_to_id)
    valid_dataset = CreateData(valid_X, valid_Y, translate_to_id)
    test_dataset = CreateData(test_X, test_Y, translate_to_id)

    W = word2vec(ids)
    device = torch.device("cuda")

    for OUT_CHANNEL in OUT_CHANNELS:
        for ALEARNING_RATE in LEARNIG_RATE:
            model = CNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, IN_CHANNELS, OUT_CHANNEL, KERNEL_HEIGHTS, STRIDE, PADDING, emb_weights=W)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=ALEARNING_RATE)

            valid_acc = train_model(train_dataset, valid_dataset, model, criterion, optimizer, NUM_EPOCHS, BATCH_SIZE, collate_fn=Padsequence(PADDING_IDX), device=device)

            train_acc, valid_acc, test_acc = result(train_dataset, test_dataset, model, criterion, device)
            if valid_acc > best_valid:
                best_train = train_acc
                best_valid = valid_acc
                best_test = test_acc
                best_outchannel = OUT_CHANNEL
                best_lr = ALEARNING_RATE
    
    print(f"best OUT_CHANNELE: {best_outchannel}\nbest LEARNIG RATE: {best_lr}\ntrain acc: {best_train}\ntrain acc: {best_valid}\ntrain acc: {best_test}")

#train loss: 0.52695, train accuracy: 0.8125233994758517, valid loss: 0.6588987112045288, valid accuracy: 0.7612275449101796, time: 151.83889sec