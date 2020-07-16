# 81. RNNによる予測
# ID番号で表現された単語列x=(x1,x2,…,xT)がある．
# ただし，Tは単語列の長さ，xt∈ℝVは単語のID番号のone-hot表記である（Vは単語の総数である）．
# 再帰型ニューラルネットワーク（RNN: Recurrent Neural Network）を用い，単語列xからカテゴリyを予測するモデルとして，次式を実装せよ．

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from collections import defaultdict
from torch import nn
import pandas as pd
import string
import torch

def tokenizer(text, word2id, unk=0):
  ids = []
  for word in text.split():
      ids.append(word2id.get(word, unk))
  return ids

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

    # モデルを定義する
    model = RNN(vocab_size=VOCAB_SIZE, padding_idx=PADDING_IDX, output_size=4)

    # 先頭10件の予測値取得
    for i in range(10):
        X = dataset_train[i]['inputs']
        print(torch.softmax(model(X.unsqueeze(0)), dim=-1))

# 結果
# tensor([[0.3831, 0.2058, 0.1417, 0.2694]], grad_fn=<SoftmaxBackward>)
# tensor([[0.2850, 0.2662, 0.1382, 0.3107]], grad_fn=<SoftmaxBackward>)
# tensor([[0.2011, 0.3507, 0.2212, 0.2271]], grad_fn=<SoftmaxBackward>)
# tensor([[0.2950, 0.2075, 0.1821, 0.3154]], grad_fn=<SoftmaxBackward>)
# tensor([[0.2031, 0.2471, 0.3398, 0.2100]], grad_fn=<SoftmaxBackward>)
# tensor([[0.3579, 0.2393, 0.2140, 0.1889]], grad_fn=<SoftmaxBackward>)
# tensor([[0.3294, 0.1377, 0.3824, 0.1504]], grad_fn=<SoftmaxBackward>)
# tensor([[0.3174, 0.1874, 0.3960, 0.0992]], grad_fn=<SoftmaxBackward>)
# tensor([[0.2831, 0.2353, 0.2341, 0.2475]], grad_fn=<SoftmaxBackward>)
# tensor([[0.3223, 0.1711, 0.4179, 0.0887]], grad_fn=<SoftmaxBackward>)