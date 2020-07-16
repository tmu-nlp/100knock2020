import pickle
import numpy as numpy
from gensim.models import KeyedVectors
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
import optuna

from knock80 import obtain_data, tokenize
from knock81 import CreateDataset
from knock82 import calculate_loss_and_acc, train_model
from knock83 import Padsequence
from knock86 import CNN

def objective(trial):
    out_channels = int(trial.suggest_discrete_uniform('out_channels', 50, 200, 50))
    drop_rate = trial.suggest_discrete_uniform('drop_rate', 0.0, 0.5, 0.1)
    learning_rate = trial.suggest_loguniform('learning_rate', 5e-4, 5e-2)
    momentum = trial.suggest_discrete_uniform('momentum', 0.5, 0.9, 0.1)
    batch_size = int(trial.suggest_discrete_uniform('batch_size', 16, 128, 16))

    VOCAB_SIZE = len(set(word_to_id.values())) + 1
    PADDING_IDX = len(set(word_to_id.values()))
    EMB_SIZE = 300
    KERNEL_HEIGHTS = 3
    STRIDE = 1
    PADDING = 1
    OUTPUT_SIZE = 4
    CONV_PARAMS = [[2, 0], [3, 1], [4, 2]]
    NUM_EPOCHS = 10

    model = CNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, OUT_CHANNELS, KERNEL_HEIGHTS, STRIDE, PADDING, emb_weights=weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    device = torch.device('cuda')

    log = train_model(dataset_train, dataset_dev, batch_size, model, criterion, optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX), device=device)

    loss_dev, _ = calculate_loss_and_acc(model, dataset_dev, criterion=criterion, device=device) 

  return loss_dev 

if __name__ == '__main__':
    dataset_train = torch.load('./dataset_train.pt')
    dataset_dev = torch.load('./dataset_dev.pt')
    word_to_id = pickle.load(open('./word_to_id.pkl', 'rb'))
    weights = torch.load('./weights.pt')

    study = optuna.create_study()
    study.optimize(objective, timeout=7200)

    print('Best trial:')
    trial = study.best_trial
    print('  Value: {:.3f}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

"""
Best trial:
  Value: 0.442
  Params: 
    out_channels: 100.0
    drop_rate: 0.30000000000000004
    learning_rate: 0.020264984164593963
    momentum: 0.5
    batch_size: 16.0

epoch: 1, loss_train: 0.6455, accuracy_train: 0.7395, loss_dev: 0.7287, accuracy_dev: 0.7111
epoch: 2, loss_train: 0.4672, accuracy_train: 0.8426, loss_dev: 0.6307, accuracy_dev: 0.7566
epoch: 3, loss_train: 0.3611, accuracy_train: 0.8718, loss_dev: 0.5308, accuracy_dev: 0.7979
epoch: 4, loss_train: 0.3134, accuracy_train: 0.8934, loss_dev: 0.5100, accuracy_dev: 0.8106
epoch: 5, loss_train: 0.2569, accuracy_train: 0.9205, loss_dev: 0.4673, accuracy_dev: 0.8222
epoch: 6, loss_train: 0.2317, accuracy_train: 0.9255, loss_dev: 0.4628, accuracy_dev: 0.8307
epoch: 7, loss_train: 0.2243, accuracy_train: 0.9249, loss_dev: 0.4656, accuracy_dev: 0.8127
epoch: 8, loss_train: 0.2069, accuracy_train: 0.9348, loss_dev: 0.4459, accuracy_dev: 0.8392
epoch: 9, loss_train: 0.2008, accuracy_train: 0.9370, loss_dev: 0.4406, accuracy_dev: 0.8392
epoch: 10, loss_train: 0.1984, accuracy_train: 0.9392, loss_dev: 0.4422, accuracy_dev: 0.8370
"""
