"""
85. 双方向RNN・多層化Permalink
順方向と逆方向のRNNの両方を用いて入力テキストをエンコードし，モデルを学習せよ．

h⃖ T+1=0,h⃖ t=RNN←−−−(emb(xt),h⃖ t+1),y=softmax(W(yh)[h→T;h⃖ 1]+b(y))

ただし，h→t∈ℝdh,h⃖ t∈ℝdhはそれぞれ，順方向および逆方向のRNNで求めた時刻tの隠れ状態ベクトル，
RNN←−−−(x,h)は入力xと次時刻の隠れ状態hから前状態を計算するRNNユニット，
W(yh)∈ℝL×2dhは隠れ状態ベクトルからカテゴリを予測するための行列，
b(y)∈ℝLはバイアス項である．
また，[a;b]はベクトルaとbの連結を表す。

さらに，双方向RNNを多層化して実験せよ．
"""

from gensim.models import KeyedVectors
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

train_path = "../chapter06/train.csv"
valid_path = "../chapter06/valid.csv"
test_path = "../chapter06/test.csv"

_, V = translate_to_id("")

VOCAB_SIZE = V+1
EMB_SIZE = 300
PADDING_IDX = V
OUTPUT_SIZE = 4
HIDDEN_SIZE = 50
NUM_LAYERS = 2
LEARNIG_RATE = 0.05
BATCH_SIZE = 32
NUM_EPOCHS = 10

with open("id_file", encoding="utf-8") as f:
    ids = {}
    for line in f:
        id, word = line.split()
        ids[word] = int(id)


model = KeyedVectors.load_word2vec_format("./GoogleNews-vectors-negative300.bin.gz", binary=True)

class RNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, hidden_size, num_layers, emb_weights=None, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = bidirectional+1   #単方向なら1, 双方向なら2
        if emb_weights != None:
            self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        #双方向はここで指定
        self.rnn = nn.RNN(emb_size, hidden_size, num_layers, nonlinearity="tanh", bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_size*self.num_directions, output_size)

    def forward(self, x):
        self.batch_size = x.size()[0]
        hidden = self.init_hidden()
        emb = self.emb(x)
        out, hidden = self.rnn(emb, hidden.to("cuda"))
        out = self.fc(out[:, -1])
        return out

    def init_hidden(self):
        hidden = torch.zeros(self.num_layers*self.num_directions, self.batch_size, self.hidden_size)
        return hidden

#全問まででidを割り振った単語に単語ベクトルを適用する
#モデルになかったらランダムで単語ベクトルを割り振る
def word2vec(ids):
    weights = np.zeros((VOCAB_SIZE, EMB_SIZE))
    words_in_pretrained = 0
    for i, word in enumerate(ids.keys()):
        try:
            weights[i] = model[word]
            words_in_pretrained += 1
        except KeyError:
            weights[i] = np.random.normal(scale=0.4, size=(EMB_SIZE, ))
    weights = torch.from_numpy(weights.astype((np.float32)))
    return weights

    #print(f"学習済ベクトル利用単語数: {words_in_pretrained} / {VOCAB_SIZE}")
    #print(weights.size())



if __name__=="__main__":
    train_X, train_Y = get_X_Y(train_path)
    valid_X, valid_Y = get_X_Y(valid_path)
    test_X, test_Y = get_X_Y(test_path)
    train_dataset = CreateData(train_X, train_Y, translate_to_id)
    valid_dataset = CreateData(valid_X, valid_Y, translate_to_id)
    test_dataset = CreateData(test_X, test_Y, translate_to_id)

    W = word2vec(ids)
    device = torch.device("cuda")

    model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, emb_weights=W, bidirectional=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNIG_RATE)


    log = train_model(train_dataset, valid_dataset, model, criterion, optimizer, NUM_EPOCHS, BATCH_SIZE, collate_fn=Padsequence(PADDING_IDX), device=device)

    plot(log)

    result(train_dataset, test_dataset, model, criterion, device)