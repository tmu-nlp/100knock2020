import time
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

from knock71 import SingleLayerPerceptron
from knock73 import CreateDataset, obtain_dataloader
from knock74 import calculate_accuracy
from knock75 import calculate_loss_and_accuracy, train_sgd_log, plot_loss_and_accuracy
from knock76 import train_sgd_with_checkpoints

def train_sgd_minibatch(train_dataset, dev_dataset, batch_size, model, criterion, optimizer, num_epochs):

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=len(dev_dataset), shuffle=False)

    train_log = []
    dev_log = []
    for epoch in range(num_epochs):
        s_time = time.time()

        model.train()
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = calculate_loss_and_accuracy(model, criterion, train_dataloader)
        dev_loss, dev_acc = calculate_loss_and_accuracy(model, criterion, dev_dataloader)
        train_log.append([train_loss, train_acc])
        dev_log.append([dev_loss, dev_acc])

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'./checkpoints/slp/batchsize{batch_size}_checkpoint{epoch + 1}.pt')

        e_time = time.time()

        print(f'epoch: {epoch + 1}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, dev_loss: {dev_loss:.4f}, dev_acc: {dev_acc:.4f}, {(e_time - s_time):.4f}sec') 

    return {'train': train_log, 'dev': dev_log}

if __name__ == '__main__':
    X_train = torch.load('./tensors/X_train')
    Y_train = torch.load('./tensors/Y_train')
    X_dev = torch.load('./tensors/X_dev')
    Y_dev = torch.load('./tensors/Y_dev')
    X_test = torch.load('./tensors/X_test')
    Y_test = torch.load('./tensors/Y_test')

    train_dataset = CreateDataset(X_train, Y_train)
    dev_dataset = CreateDataset(X_dev, Y_dev)

    model = SingleLayerPerceptron(300, 4)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    for batch_size in [2 ** i for i in range(11)]:
        print(f'batch_size: {batch_size}')
        log = train_sgd_minibatch(train_dataset, dev_dataset, batch_size, model, criterion, optimizer, 1)

"""
batch_size: 1
epoch: 1, train_loss: 0.4519, train_acc: 0.8434, dev_loss: 0.4711, dev_acc: 0.8328, 2.3987sec
batch_size: 2
epoch: 1, train_loss: 0.4142, train_acc: 0.8567, dev_loss: 0.4443, dev_acc: 0.8508, 1.6871sec
batch_size: 4
epoch: 1, train_loss: 0.4011, train_acc: 0.8607, dev_loss: 0.4367, dev_acc: 0.8635, 1.1314sec
batch_size: 8
epoch: 1, train_loss: 0.3951, train_acc: 0.8633, dev_loss: 0.4329, dev_acc: 0.8614, 0.5040sec
batch_size: 16
epoch: 1, train_loss: 0.3924, train_acc: 0.8636, dev_loss: 0.4317, dev_acc: 0.8593, 0.2840sec
batch_size: 32
epoch: 1, train_loss: 0.3912, train_acc: 0.8637, dev_loss: 0.4310, dev_acc: 0.8614, 0.2230sec
batch_size: 64
epoch: 1, train_loss: 0.3906, train_acc: 0.8643, dev_loss: 0.4307, dev_acc: 0.8624, 0.1926sec
batch_size: 128
epoch: 1, train_loss: 0.3904, train_acc: 0.8643, dev_loss: 0.4306, dev_acc: 0.8624, 0.1778sec
batch_size: 256
epoch: 1, train_loss: 0.3898, train_acc: 0.8643, dev_loss: 0.4305, dev_acc: 0.8624, 0.1672sec
batch_size: 512
epoch: 1, train_loss: 0.3901, train_acc: 0.8643, dev_loss: 0.4305, dev_acc: 0.8624, 0.1454sec
batch_size: 1024
epoch: 1, train_loss: 0.3953, train_acc: 0.8645, dev_loss: 0.4304, dev_acc: 0.8624, 0.1532sec
"""

