"""
84. 単語ベクトルの導入Permalink
事前学習済みの単語ベクトル（例えば，Google Newsデータセット（約1,000億単語）での学習済み単語ベクトル）で単語埋め込みemb(x)を初期化し，学習せよ
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
LEARNIG_RATE = 0.05
BATCH_SIZE = 32
NUM_EPOCHS = 10

with open("id_file", encoding="utf-8") as f:
    ids = {}
    for line in f:
        id, word = line.split()
        ids[word] = int(id)

model = KeyedVectors.load_word2vec_format("./GoogleNews-vectors-negative300 (1).bin.gz", binary=True)

class RNN(nn.Module):           #初期値を引数に追加
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, hidden_size, emb_weights=None):
        super().__init__()
        torch.manual_seed(7)
        self.hidden_size = hidden_size
        if emb_weights != None:     #初期値が与えられていればそれに従ってembedding
            self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.rnn = nn.RNN(emb_size, hidden_size, nonlinearity="tanh", batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        self.batch_size = x.size()[0]
        hidden = self.init_hidden()
        emb = self.emb(x)
        out, hidden = self.rnn(emb, hidden.to("cuda"))
        out = self.fc(out[:, -1])
        return out

    def init_hidden(self):
        hidden = torch.zeros(1, self.batch_size, self.hidden_size)
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

    model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE, emb_weights=W).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNIG_RATE)

    log = train_model(train_dataset, valid_dataset, model, criterion, optimizer, NUM_EPOCHS, BATCH_SIZE, collate_fn=Padsequence(PADDING_IDX), device=device)

    plot(log)

    result(train_dataset, test_dataset, criterion, device)