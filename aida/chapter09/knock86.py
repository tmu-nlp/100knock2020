import pickle
import numpy as numpy
from gensim.models import KeyedVectors
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

from knock80 import obtain_data, tokenize
from knock81 import CreateDataset
from knock82 import calculate_loss_and_acc, train_model
from knock83 import Padsequence

class CNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, out_channels, kernel_heights, stride, padding, emb_weights=None):
        super().__init__()
        if emb_weights != None:
            self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.conv = nn.Conv2d(1, out_channels, (kernel_heights, emb_size), stride, (padding, 0))
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(out_channels, output_size)

    def forward(self, x):
        emb = self.emb(x).unsqueeze(1)
        conv = self.conv(emb)
        act = F.relu(conv.squeeze(3))
        max_pool = F.max_pool1d(act, act.size()[2])
        out = self.fc(self.drop(max_pool.squeeze(2)))
        return out

if __name__ == '__main__':
    dataset_train = torch.load('./dataset_train.pt')
    dataset_dev = torch.load('./dataset_dev.pt')
    word_to_id = pickle.load(open('./word_to_id.pkl', 'rb'))
    weights = torch.load('./weights.pt')

    VOCAB_SIZE = len(set(word_to_id.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(word_to_id.values()))
    OUTPUT_SIZE = 4
    OUT_CHANNELS = 100
    KERNEL_HEIGHTS = 3
    STRIDE = 1
    PADDING = 1

    cnn = CNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, OUT_CHANNELS, KERNEL_HEIGHTS, STRIDE, PADDING, emb_weights=weights)

    for i in range(10):
        X = dataset_train[i]['inputs']
        print(f'{i}-th input')
        print(f'  input: {X}')
        print(f'  output: {torch.softmax(cnn(X.unsqueeze(0)), dim=-1)}')

"""
0-th input
  input: tensor([   3, 1662,   24, 1443,  922,  132,  192,    0, 1278,    7])
  output: tensor([[0.1735, 0.1793, 0.3137, 0.3336]], grad_fn=<SoftmaxBackward>)
1-th input
  input: tensor([   0,  237,    0, 4401, 4402,  366,  170,    0,    1])
  output: tensor([[0.2243, 0.1641, 0.1465, 0.4650]], grad_fn=<SoftmaxBackward>)
2-th input
  input: tensor([   0,  266, 3341,  138, 1141, 2678])
  output: tensor([[0.2467, 0.2308, 0.2498, 0.2727]], grad_fn=<SoftmaxBackward>)
3-th input
  input: tensor([  5, 399, 830,  59,  74, 755,  94])
  output: tensor([[0.2675, 0.3061, 0.1804, 0.2460]], grad_fn=<SoftmaxBackward>)
4-th input
  input: tensor([  68, 2276,  462,    0,    0,  101, 2277])
  output: tensor([[0.2109, 0.1411, 0.2714, 0.3766]], grad_fn=<SoftmaxBackward>)
5-th input
  input: tensor([   0, 4403, 1663,    0, 2679,    0,    0, 4404, 4405,    1])
  output: tensor([[0.1956, 0.1694, 0.3110, 0.3240]], grad_fn=<SoftmaxBackward>)
6-th input
  input: tensor([2680,  367, 4406,   33,  346, 3342, 2278, 1664,  238,   23])
  output: tensor([[0.2317, 0.2563, 0.2505, 0.2616]], grad_fn=<SoftmaxBackward>)
7-th input
  input: tensor([ 181,  463,    4, 1142, 3343, 1917, 3344,    1])
  output: tensor([[0.2676, 0.1951, 0.2689, 0.2684]], grad_fn=<SoftmaxBackward>)
8-th input
  input: tensor([   0,    0,    0,  171,    0, 4407,    0,  693,  226,    1])
  output: tensor([[0.2162, 0.1604, 0.2112, 0.4122]], grad_fn=<SoftmaxBackward>)
9-th input
  input: tensor([   5,    4,   14,   15, 3345, 4408,    0,   22])
  output: tensor([[0.2134, 0.1916, 0.2841, 0.3109]], grad_fn=<SoftmaxBackward>)
"""