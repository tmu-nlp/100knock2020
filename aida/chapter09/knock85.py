import pickle
import numpy as numpy
from gensim.models import KeyedVectors
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from knock80 import obtain_data, tokenize
from knock81 import CreateDataset
from knock82 import calculate_loss_and_acc, train_model
from knock83 import Padsequence
from knock84 import RNN

if __name__ == '__main__':
    dataset_train = torch.load('./dataset_train.pt')
    dataset_dev = torch.load('./dataset_dev.pt')
    word_to_id = pickle.load(open('./word_to_id.pkl', 'rb'))

    VOCAB_SIZE = len(set(word_to_id.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(word_to_id.values()))
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50
    NUM_LAYERS = 2
    LEARNING_RATE = 5e-2
    BATCH_SIZE = 32
    NUM_EPOCHS = 10

    model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, emb_weights=weights, bidirectional=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    device = torch.device('cuda')

    log = train_model(dataset_train, dataset_dev, BATCH_SIZE, model, criterion, 
                      optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX), device=device)

"""
epoch: 1, loss_train: 1.1563, accuracy_train: 0.5491, loss_dev: 1.1570, accuracy_dev: 0.5503
epoch: 2, loss_train: 0.8923, accuracy_train: 0.6708, loss_dev: 0.9272, accuracy_dev: 0.6677
epoch: 3, loss_train: 0.7230, accuracy_train: 0.7391, loss_dev: 0.7517, accuracy_dev: 0.7333
epoch: 4, loss_train: 0.9952, accuracy_train: 0.6790, loss_dev: 1.1319, accuracy_dev: 0.6508
epoch: 5, loss_train: 0.6714, accuracy_train: 0.7521, loss_dev: 0.7194, accuracy_dev: 0.7397
epoch: 6, loss_train: 0.7350, accuracy_train: 0.7252, loss_dev: 0.7951, accuracy_dev: 0.7079
epoch: 7, loss_train: 0.6912, accuracy_train: 0.7491, loss_dev: 0.7422, accuracy_dev: 0.7344
epoch: 8, loss_train: 0.6084, accuracy_train: 0.7701, loss_dev: 0.6719, accuracy_dev: 0.7492
epoch: 9, loss_train: 0.6045, accuracy_train: 0.7726, loss_dev: 0.6699, accuracy_dev: 0.7492
epoch: 10, loss_train: 0.6024, accuracy_train: 0.7736, loss_dev: 0.6681, accuracy_dev: 0.7481
"""