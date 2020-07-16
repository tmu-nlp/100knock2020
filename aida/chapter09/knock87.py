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
from knock86 import CNN

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
    LEARNING_RATE = 5e-2
    BATCH_SIZE = 64
    NUM_EPOCHS = 10

    cnn = CNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, OUT_CHANNELS, KERNEL_HEIGHTS, STRIDE, PADDING, emb_weights=weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    device = torch.device('cuda')

    log = train_model(dataset_train, dataset_dev, BATCH_SIZE, model, criterion, 
                      optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX), device=device)

"""
epoch: 1, loss_train: 0.9163, accuracy_train: 0.5920, loss_dev: 0.9490, accuracy_dev: 0.5810
epoch: 2, loss_train: 0.7083, accuracy_train: 0.7295, loss_dev: 0.7818, accuracy_dev: 0.7048
epoch: 3, loss_train: 0.6028, accuracy_train: 0.7655, loss_dev: 0.7016, accuracy_dev: 0.7312
epoch: 4, loss_train: 0.5449, accuracy_train: 0.7975, loss_dev: 0.6676, accuracy_dev: 0.7460
epoch: 5, loss_train: 0.4979, accuracy_train: 0.8138, loss_dev: 0.6348, accuracy_dev: 0.7587
epoch: 6, loss_train: 0.4624, accuracy_train: 0.8274, loss_dev: 0.5948, accuracy_dev: 0.7714
epoch: 7, loss_train: 0.4939, accuracy_train: 0.7995, loss_dev: 0.6489, accuracy_dev: 0.7397
epoch: 8, loss_train: 0.4311, accuracy_train: 0.8454, loss_dev: 0.5769, accuracy_dev: 0.7862
epoch: 9, loss_train: 0.4169, accuracy_train: 0.8597, loss_dev: 0.5694, accuracy_dev: 0.7958
epoch: 10, loss_train: 0.4151, accuracy_train: 0.8605, loss_dev: 0.5675, accuracy_dev: 0.7968
"""