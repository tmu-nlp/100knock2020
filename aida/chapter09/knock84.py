import pickle
import numpy as np
from gensim.models import KeyedVectors
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from knock80 import obtain_data, tokenize
from knock81 import CreateDataset
from knock82 import calculate_loss_and_acc, train_model
from knock83 import Padsequence

class RNN(nn.Module):
  """ some changes from knock81's RNN
   - multi layer
   - adopt bidirectional
   - initialize emb with pre-trained vectors
  """
  def __init__(self, vocab_size, emb_size, padding_idx, output_size, hidden_size, num_layers, emb_weights=None, bidirectional=False):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.num_directions = bidirectional + 1
    if emb_weights != None:
      self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
    else:
      self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
    self.rnn = nn.RNN(emb_size, hidden_size, num_layers, nonlinearity='tanh', bidirectional=bidirectional, batch_first=True)
    self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

  def forward(self, x):
    self.batch_size = x.size()[0]
    hidden = self.init_hidden()
    emb = self.emb(x)
    out, hidden = self.rnn(emb, hidden)
    out = self.fc(out[:, -1, :])
    return out

  def init_hidden(self):
    hidden = torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size)
    return hidden

if __name__ == '__main__':
    w2v = KeyedVectors.load_word2vec_format('../chapter07/GoogleNews-vectors-negative300.bin.gz', binary=True)
    word_to_id = pickle.load(open('./word_to_id.pkl', 'rb'))
    dataset_train = torch.load('./dataset_train.pt')
    dataset_dev = torch.load('./dataset_dev.pt')

    VOCAB_SIZE = len(set(word_to_id.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(word_to_id.values()))
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50
    NUM_LAYERS = 1
    LEARNING_RATE = 5e-2
    BATCH_SIZE = 32
    NUM_EPOCHS = 10

    weights = np.random.normal(size=(VOCAB_SIZE, EMB_SIZE))
    c_w2v = 0
    for word, id in word_to_id.items():
        if word in w2v.wv.vocab and id > 0:
            c_w2v += 1
            weights[id] = w2v[word]
    weights = torch.from_numpy(weights.astype(np.float32))
    torch.save(weights, './weights.pt')
    print(f'use {c_w2v} w2v vectors in {VOCAB_SIZE} words')
    exit()

    model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, emb_weights=weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    device = torch.device('cuda')
    log = train_model(dataset_train, dataset_dev, BATCH_SIZE, model, criterion,
                      optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX), device=device)
    
"""
use 5103 w2v vectors in 6814 words

epoch: 1, loss_train: 1.2335, accuracy_train: 0.4849, loss_dev: 1.2567, accuracy_dev: 0.4646
epoch: 2, loss_train: 1.1873, accuracy_train: 0.5236, loss_dev: 1.2014, accuracy_dev: 0.5143
epoch: 3, loss_train: 1.1256, accuracy_train: 0.5601, loss_dev: 1.1380, accuracy_dev: 0.5534
epoch: 4, loss_train: 0.9398, accuracy_train: 0.6690, loss_dev: 0.9743, accuracy_dev: 0.6519
epoch: 5, loss_train: 0.8846, accuracy_train: 0.6830, loss_dev: 0.9155, accuracy_dev: 0.6720
epoch: 6, loss_train: 0.8378, accuracy_train: 0.7003, loss_dev: 0.8839, accuracy_dev: 0.6825
epoch: 7, loss_train: 0.7939, accuracy_train: 0.7109, loss_dev: 0.8525, accuracy_dev: 0.6857
epoch: 8, loss_train: 0.7825, accuracy_train: 0.7159, loss_dev: 0.8457, accuracy_dev: 0.6899
epoch: 9, loss_train: 0.7578, accuracy_train: 0.7231, loss_dev: 0.8049, accuracy_dev: 0.6995
epoch: 10, loss_train: 0.7483, accuracy_train: 0.7264, loss_dev: 0.7951, accuracy_dev: 0.7037
"""