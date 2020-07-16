# 83. ミニバッチ化・GPU上での学習
# 問題82のコードを改変し，B事例ごとに損失・勾配を計算して学習を行えるようにせよ（Bの値は適当に選べ）．
# また，GPU上で学習を実行せよ．

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from collections import defaultdict
from torch import optim
from torch import nn
import pandas as pd
import numpy as np
import string
import torch
import time

def tokenizer(text, word2id, unk=0):
  ids = []
  for word in text.split():
      ids.append(word2id.get(word, unk))
  return ids

def calculate_loss_and_accuracy(model, dataset, device=None, criterion=None):
  dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
  loss = 0
  total = 0
  correct = 0
  with torch.no_grad():
    for data in dataloader:
      inputs = data['inputs'].to(device)
      labels = data['labels'].to(device)
      outputs = model(inputs)
      # 損失を加算する
      if criterion != None:
        loss += criterion(outputs, labels).item()
      # 正解数を数える
      pred = torch.argmax(outputs, dim=-1)
      correct += (pred == labels).sum().item()
      total += len(inputs)

  return loss / len(dataset), correct / total


def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, collate_fn=None, device=None):
  # デバイスの指定
  #model.to(device)
  # dataloaderの作成
  dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
  dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False)
  # スケジューラの設定
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-5, last_epoch=-1)

  for epoch in range(num_epochs):
    model.train()
    for data in dataloader_train:
      optimizer.zero_grad()
      inputs = data['inputs'].to(device)
      labels = data['labels'].to(device)
      outputs = model.forward(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
    model.eval()
    # 損失と正解率を算出する
    loss_train, acc_train = calculate_loss_and_accuracy(model, dataset_train, device, criterion=criterion)
    loss_valid, acc_valid = calculate_loss_and_accuracy(model, dataset_valid, device, criterion=criterion)
    # ログを出力する
    print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}')
    scheduler.step()

class Padsequence():
  def __init__(self, padding_idx):
    self.padding_idx = padding_idx

  def __call__(self, batch):
    sorted_batch = sorted(batch, key=lambda x: x['inputs'].shape[0], reverse=True)
    sequences = [x['inputs'] for x in sorted_batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.padding_idx)
    labels = torch.LongTensor([x['labels'] for x in sorted_batch])
    call = {}
    call["inputs"] = sequences_padded
    call["labels"] = labels
    return call

class CreateDataset(Dataset):
  def __init__(self, X, y, tokenizer, word2id):
    self.X = X
    self.y = y
    self.tokenizer = tokenizer

  def __len__(self):  # len(Dataset)で返す値を指定
    return len(self.y)

  def __getitem__(self, index):  # Dataset[index]で返す値を指定
    text = self.X[index]
    inputs = self.tokenizer(text=text, word2id=word2id)
    item = {}
    item["inputs"] = torch.tensor(inputs, dtype=torch.int64)
    item["labels"] = torch.tensor(self.y[index], dtype=torch.int64)
    return item

class RNN(nn.Module):
  def __init__(self, vocab_size, emb_size=300, padding_idx=0, output_size=1, hidden_size=50):
    super().__init__()
    self.hidden_size = hidden_size
    self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
    self.rnn = nn.RNN(emb_size, hidden_size, nonlinearity='tanh', batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    self.batch_size = x.size()[0]
    hidden = self.init_hidden()
    emb = self.emb(x)
    out, hidden = self.rnn(emb, hidden)
    out = self.fc(out[:, -1, :])
    return out

  def init_hidden(self):
    hidden = torch.zeros(1, self.batch_size, self.hidden_size)
    return hidden

if __name__ == "__main__":
    # データを読み込む
    train = pd.read_csv('train.txt', header=None, sep='\t', names=['TITLE', 'CATEGORY'])
    valid = pd.read_csv('valid.txt', header=None, sep='\t', names=['TITLE', 'CATEGORY'])
    test = pd.read_csv('test.txt', header=None, sep='\t', names=['TITLE', 'CATEGORY'])

    # 単語の頻度を集計する
    d = defaultdict(int)
    for text in train['TITLE']:
        for word in text.split():
            d[word] += 1
    d = sorted(d.items(), key=lambda x:x[1], reverse=True)

    # 単語ID辞書を作成する
    word2id = {}
    for i, (word, cnt) in enumerate(d):
        # 出現頻度が2回以上の単語を登録する
        if cnt <= 1:
            continue
        word2id[word] = i + 1

    # ラベルを数値に変換する
    category_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
    y_train = train['CATEGORY'].map(lambda x: category_dict[x]).values
    y_valid = valid['CATEGORY'].map(lambda x: category_dict[x]).values
    y_test = test['CATEGORY'].map(lambda x: category_dict[x]).values

    # Datasetを作成する
    dataset_train = CreateDataset(train['TITLE'], y_train, tokenizer, word2id)
    dataset_valid = CreateDataset(valid['TITLE'], y_valid, tokenizer, word2id)
    dataset_test = CreateDataset(test['TITLE'], y_test, tokenizer, word2id)

    # パラメータを設定する
    VOCAB_SIZE = len(set(word2id.values())) + 1
    PADDING_IDX = len(set(word2id.values()))
    BATCH_SIZE = 32
    NUM_EPOCHS = 3

    # モデルを定義する
    model = RNN(vocab_size=VOCAB_SIZE, padding_idx=PADDING_IDX, output_size=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # デバイスの指定
    #device = torch.device('cuda')

    # モデルを学習する
    train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX))

    # 正解率を表示する
    _, acc_train = calculate_loss_and_accuracy(model, dataset_train)
    _, acc_test = calculate_loss_and_accuracy(model, dataset_test)
    print(f'train acc: {acc_train}')
    print(f'test acc: {acc_test}')

# 結果
# epoch: 1, loss_train: 1.2551, accuracy_train: 0.4123, loss_valid: 1.2368, accuracy_valid: 0.4175
# epoch: 2, loss_train: 1.2586, accuracy_train: 0.4206, loss_valid: 1.2403, accuracy_valid: 0.4145
# epoch: 3, loss_train: 1.2582, accuracy_train: 0.4263, loss_valid: 1.2414, accuracy_valid: 0.4213
# train acc: 0.426
# test acc: 0.425