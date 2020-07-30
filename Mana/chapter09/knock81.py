# ID番号で表現された単語列x=(x1,x2,…,xT)がある．
# ただし，Tは単語列の長さ，xt∈ℝVは単語のID番号のone-hot表記である（Vは単語の総数である）．
# 再帰型ニューラルネットワーク（RNN: Recurrent Neural Network）を用い，
# 単語列xからカテゴリyを予測するモデルとして，次式を実装せよ．
# https://qiita.com/yamaru/items/c5f87d55d00066f8ad7c#81-rnn%E3%81%AB%E3%82%88%E3%82%8B%E4%BA%88%E6%B8%AC


import torch 
from torch import nn

class RNN(nn.Module):
  def __init__(self, vocab_size, emb_size, padding_idx, output_size, hidden_size):
    super().__init__()
    self.hidden_size = hidden_size
    self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
    self.rnn = nn.RNN(emb_size, hidden_size, nonlinearity='tanh', batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    self.batch_size = x.size()[0]
    hidden = self.init_hidden()  # h0のゼロベクトルを作成
    emb = self.emb(x)
    # emb.size() = (batch_size, seq_len, emb_size)
    out, hidden = self.rnn(emb, hidden)
    # out.size() = (batch_size, seq_len, hidden_size)
    out = self.fc(out[:, -1, :])
    # out.size() = (batch_size, output_size)
    return out

  def init_hidden(self):
    hidden = torch.zeros(1, self.batch_size, self.hidden_size)
    return hidden

from torch.utils.data import Dataset

class CreateDataset(Dataset):
  def __init__(self, X, y, ids):
    self.X = X
    self.y = y
    self.ids = ids

  def __len__(self):  # len(Dataset)で返す値を指定
    return len(self.y)

  def __getitem__(self, index):  # Dataset[index]で返す値を指定
    text = self.X[index]
    inputs = preprocess(text)
    inputs = sent2ids(inputs, self.ids)

    return {
      'inputs': torch.tensor(inputs, dtype=torch.int64),
      'labels': torch.tensor(self.y[index], dtype=torch.int64)
    }

# Datasetの作成
dataset_train = CreateDataset(X_train['Title'].values, y_train, ids)
dataset_valid = CreateDataset(X_valid['Title'].values, y_valid, ids)
dataset_test = CreateDataset(X_test['Title'].values, y_test, ids)

# パラメータの設定
VOCAB_SIZE = len(set(ids.values())) + 1  # 辞書のID数 + パディングID
EMB_SIZE = 300
PADDING_IDX = len(set(ids.values()))
OUTPUT_SIZE = 4
HIDDEN_SIZE = 50

# モデルの定義
model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE)

# 先頭10件の予測値取得
for i in range(10):
  X = dataset_train[i]['inputs']
  print(torch.softmax(model(X.unsqueeze(0)), dim=-1))