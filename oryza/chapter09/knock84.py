from gensim.models import KeyedVectors
from torch.utils.data import DataLoader
from knock81 import CreateDataset
from knock80 import get_feature, word2ids, get_label
from knock82 import calculate_loss_and_accuracy, train_model
from knock83 import Padsequence
import joblib
import time
import torch
from torch import optim, nn
import numpy as np

class RNN(nn.Module):
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

if __name__ == "__main__":
    x_train = get_feature(open('train2.feature.txt'))
    y_train = joblib.load('train_label.pkl').tolist()
    x_valid = get_feature(open('valid2.feature.txt'))
    y_valid = joblib.load('valid_label.pkl').tolist()
    x_test = get_feature(open('test2.feature.txt'))
    y_test = joblib.load('test_label.pkl').tolist()

    tr_text2id, tr_word2id = word2ids(x_train)
    train_set = CreateDataset(x_train, y_train, tr_text2id)
    val_text2id, val_word2id = word2ids(x_valid)
    valid_set = CreateDataset(x_valid, y_valid, val_text2id)
    test_text2id, test_word2id = word2ids(x_test)
    test_set = CreateDataset(x_test, y_test, test_text2id)

    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    VOCAB_SIZE = len(set(tr_word2id.values())) + 1
    EMB_SIZE = 300
    weights = np.zeros((VOCAB_SIZE, EMB_SIZE))
    words_in_pretrained = 0
    for i, word in enumerate(tr_word2id.keys()):
        try:
            weights[i] = model[word]
            words_in_pretrained += 1
        except KeyError:
            weights[i] = np.random.normal(scale=0.4, size=(EMB_SIZE,))
    weights = torch.from_numpy(weights.astype((np.float32)))
    
    # print(weights)

    VOCAB_SIZE = len(set(tr_word2id.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(tr_word2id.values()))
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50
    NUM_LAYERS = 1
    LEARNING_RATE = 5e-2
    BATCH_SIZE = 32
    NUM_EPOCHS = 10

    model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, emb_weights=weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    log = train_model(train_set, valid_set, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX))