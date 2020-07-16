import pickle
import torch
from torch import nn
from torch.utils.data import Dataset

from knock80 import obtain_data, tokenize

class CreateDataset(Dataset):
  def __init__(self, X, y, id_to_word, tokenize):
    self.X = X
    self.y = y
    self.id_to_word = id_to_word
    self.tokenize = tokenize

  def __len__(self):
    return len(self.y)

  def __getitem__(self, index):
    words = self.X[index]
    inputs = self.tokenize(words, self.id_to_word)

    return {
      'inputs': torch.tensor(inputs, dtype=torch.int64),
      'labels': torch.tensor(self.y[index], dtype=torch.int64)
    }

class RNN(nn.Module):
  def __init__(self, vocab_size, emb_size, padding_idx, output_size, hidden_size):
    super().__init__()
    self.hidden_size = hidden_size
    self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
    self.rnn = nn.RNN(emb_size, hidden_size, nonlinearity='tanh', batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    self.batch_size = x.size()[0]
    hidden = self.init_hidden()
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

if __name__ == '__main__':
  file_train = './../chapter06/train.feature.txt'
  file_dev = './../chapter06/dev.feature.txt'
  file_test = './../chapter06/test.feature.txt'
  X_train, y_train, _ = obtain_data(file_train)
  X_dev, y_dev, _ = obtain_data(file_dev)
  X_test, y_test, _ = obtain_data(file_test)

  word_to_id = pickle.load(open('./word_to_id.pkl', 'rb'))

  dataset_train = CreateDataset(X_train, y_train, word_to_id, tokenize)
  dataset_dev = CreateDataset(X_dev, y_dev, word_to_id, tokenize)
  dataset_test = CreateDataset(X_test, y_test, word_to_id, tokenize)
  torch.save(dataset_train, './dataset_train.pt')
  torch.save(dataset_dev, './dataset_dev.pt')
  torch.save(dataset_test, './dataset_test.pt')

  print(f'train dataset: {len(dataset_train)}')
  for i in range(5):
    for var in dataset_train[i]:
      print(f'  {var}: {dataset_train[i][var]}')

  VOCAB_SIZE = len(set(word_to_id.values())) + 1
  EMB_SIZE = 300
  PADDING_IDX = len(set(word_to_id.values()))
  OUTPUT_SIZE = 4
  HIDDEN_SIZE = 50

  rnn = RNN(vocab_size=VOCAB_SIZE, emb_size=EMB_SIZE, padding_idx=PADDING_IDX,
            output_size=OUTPUT_SIZE, hidden_size=HIDDEN_SIZE)
  for i in range(5):
    inputs = dataset_train[i]['inputs']
    print(f'{i}-th dataset')
    print(f'  input: {inputs}')
    print(f'  output: {torch.softmax(rnn(inputs.unsqueeze(0)), dim=-1)}')

"""
train dataset: 7561
  inputs: tensor([   3, 1662,   24, 1443,  922,  132,  192,    0, 1278,    7])
  labels: 0
  inputs: tensor([   0,  237,    0, 4401, 4402,  366,  170,    0,    1])
  labels: 3
  inputs: tensor([   0,  266, 3341,  138, 1141, 2678])
  labels: 0
  inputs: tensor([  5, 399, 830,  59,  74, 755,  94])
  labels: 1
  inputs: tensor([  68, 2276,  462,    0,    0,  101, 2277])
  labels: 3
0-th dataset
  input: tensor([   3, 1662,   24, 1443,  922,  132,  192,    0, 1278,    7])
  output: tensor([[0.3492, 0.2253, 0.1987, 0.2269]], grad_fn=<SoftmaxBackward>)
1-th dataset
  input: tensor([   0,  237,    0, 4401, 4402,  366,  170,    0,    1])
  output: tensor([[0.3894, 0.1925, 0.1518, 0.2663]], grad_fn=<SoftmaxBackward>)
2-th dataset
  input: tensor([   0,  266, 3341,  138, 1141, 2678])
  output: tensor([[0.1369, 0.5109, 0.1890, 0.1633]], grad_fn=<SoftmaxBackward>)
3-th dataset
  input: tensor([  5, 399, 830,  59,  74, 755,  94])
  output: tensor([[0.3620, 0.2746, 0.1703, 0.1930]], grad_fn=<SoftmaxBackward>)
4-th dataset
  input: tensor([  68, 2276,  462,    0,    0,  101, 2277])
  output: tensor([[0.1941, 0.2526, 0.2069, 0.3464]], grad_fn=<SoftmaxBackward>)
train dataset: 61008
  inputs: tensor([   3, 1662,   24, 1443,  922,  132,  192,    0, 1278,    7])
  labels: 0
"""
