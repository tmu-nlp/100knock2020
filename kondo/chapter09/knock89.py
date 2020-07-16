"""
89. 事前学習済み言語モデルからの転移学習Permalink
事前学習済み言語モデル（例えばBERTなど）を出発点として，ニュース記事見出しをカテゴリに分類するモデルを構築せよ．
"""

import pandas as pd
import numpy as np
import docutils.transforms.universal
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch import optim
from torch import cuda
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
import logging

logging.basicConfig(level=logging.ERROR)

train = "../chapter06/train.txt"
valid = "../chapter06/valid.txt"
test = "../chapter06/test.txt"

train_data = pd.read_csv(train, sep="\t")
valid_data = pd.read_csv(valid, sep="\t")
test_data = pd.read_csv(test, sep="\t")

class CreateDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_len):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        text = self.X[index]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "id_text": torch.LongTensor(ids),
            "mask": torch.LongTensor(mask),
            "labels": torch.Tensor(self.y[index])
        }

class BERTClass(torch.nn.Module):
    def __init__(self, drop_rate, output_size):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.drop = torch.nn.Dropout(drop_rate)
        #BERTの出力に合わせて768次元
        self.fc = torch.nn.Linear(768, output_size)

    def forward(self, ids, mask):
        _, out = self.bert(ids, attention_mask=mask)
        out = self.fc(self.drop(out))
        return out

def train_model(tr_data, va_data, model, criterion, optimizer, epochs=10, batch_size=1, collate_fn=None, device=None):

    model.to(device)

    #collate_fnはミニバッチ化する時にサイズを揃える時に使える。基本はtensorを返すだけ
    tr_l_data = DataLoader(tr_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    va_l_data = DataLoader(va_data, batch_size=1, shuffle=False)

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
            mask = X["mask"].to(device)
            labels = X["labels"].to(device)

            outputs = model.forward(inputs, mask)  #順伝播
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

    return {"train": train_log, "valid": valid_log}

def calculate_loss_acc(data, model, criterion, device=None):
    l_data = DataLoader(data, batch_size=1, shuffle=False)
    loss = 0
    total = 0
    cor = 0
    with torch.no_grad():
        for X in l_data:
            #デバイスの指定
            inputs = X["id_text"].to(device)
            mask = X["mask"].to(device)
            labels = X["labels"].to(device)

            outputs = model.forward(inputs, mask)
            loss += criterion(outputs, labels).item()

            pred = torch.argmax(outputs, dim=-1).cpu().numpy()
            labels = torch.argmax(labels, dim=-1).cpu().numpy()
            total += len(inputs)
            cor += (pred == labels).sum().item()
    return loss/len(l_data), cor/total

def plot(log):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(np.array(log["train"]).T[0], label="train")
    ax[0].plot(np.array(log["valid"]).T[0], label="valid")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].legend()
    ax[1].plot(np.array(log["train"]).T[1], label="train")
    ax[1].plot(np.array(log["valid"]).T[1], label="valid")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].legend()
    plt.show()

def result(train_dataset, test_dataset, model, criterion, device):
    _, train_acc = calculate_loss_acc(train_dataset, model, criterion, device)
    _, test_acc = calculate_loss_acc(test_dataset, model, criterion, device)
    print(f"train accuracy: {train_acc}")
    print(f"test accuracy: {test_acc}")


if __name__ == "__main__":
    DROP_RATE = 0.4
    OUTPUT_SIZE = 4
    BATCH_SIZE = 32
    NUM_EPOCHS = 4
    LEARNING_RATE = 0.00002

    #正解ラベルをone-hotベクトル化する
    y_train = pd.get_dummies(train_data, columns=["CATEGORY"])[["CATEGORY_b", "CATEGORY_e", "CATEGORY_t", "CATEGORY_m"]].values
    y_valid = pd.get_dummies(valid_data, columns=["CATEGORY"])[["CATEGORY_b", "CATEGORY_e", "CATEGORY_t", "CATEGORY_m"]].values
    y_test = pd.get_dummies(test_data, columns=["CATEGORY"])[["CATEGORY_b", "CATEGORY_e", "CATEGORY_t", "CATEGORY_m"]].values
    y_train
    """
    y_train
    [[1 0 0 0]
    [0 0 1 0]
    [0 0 0 1]
    ...
    [1 0 0 0]
    [0 0 0 1]
    [0 1 0 0]]
    """

    max_len = 20
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation=True)
    dataset_train = CreateDataset(train_data["TITLE"], y_train, tokenizer, max_len)
    dataset_valid = CreateDataset(valid_data["TITLE"], y_valid, tokenizer, max_len)
    dataset_test = CreateDataset(test_data["TITLE"], y_test, tokenizer, max_len)

    """
    for var in dataset_train[0]:
        print(f"{var}: {dataset_train[0][var]}")
    #各単語をidに変換している
    ids: tensor([  101, 25416,  9463,  1011, 10651,  1015,  1011,  2647,  2482,  4341,
            2039,  2005,  4369,  3204,  2004, 18730,  8980,   102,     0,     0])
    #0のところがpaddingされている部分
    mask: tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    #labelはone-hot
    labels: tensor([1., 0., 0., 0.])
    """

    model = BERTClass(DROP_RATE, OUTPUT_SIZE)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)

    device = "cuda"

    log = train_model(dataset_train, dataset_valid, model, criterion, optimizer, NUM_EPOCHS, BATCH_SIZE, device=device)
    plot(log)
    result(dataset_train, dataset_valid, model, criterion, device)
