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
from knock77 import train_sgd_minibatch

def calculate_loss_and_accuracy_with_gpu(model, criterion, loader, device):
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()

    return loss / len(loader), correct / total

def train_sgd_with_gpu(train_dataset, dev_dataset, batch_size, model, criterion, optimizer, num_epochs, device=None, model_name):

    model.to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=len(dataset_valid), shuffle=False)

    train_log = []
    dev_log = []
    for epoch in range(num_epochs):
        s_time = time.time()

        model.train()
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = calculate_loss_and_accuracy_with_gpu(model, criterion, train_dataloader, device=device)
        dev_loss, dev_acc = calculate_loss_and_accuracy_with_gpu(model, criterion, dev_dataloader, device=device)
        train_log.append([train_loss, train_acc])
        dev_log.append([dev_loss, dev_acc])

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'./checkpoints/{model_name}/batchsize{batch_size}_checkpoint{epoch + 1}.pt')

        e_time = time.time()

        print(f'epoch: {epoch + 1}, loss_train: {train_loss:.4f}, accuracy_train: {train_acc:.4f}, loss_valid: {dev_loss:.4f}, accuracy_valid: {dev_acc:.4f}, {(e_time - s_time):.4f}sec') 
  torch.save(model, './MLP.pt')
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
    device = torch.device('cuda')
    for batch_size in [2 ** i for i in range(11)]:
        print(f'batch_size: {batch_size}')
        log = train_sgd_with_gpu(train_dataset, dev_dataset, batch_size, model, criterion, optimizer, 1, device=device, model_name='slp')

"""
batch_size: 1
epoch: 1, loss_train: 0.4519, accuracy_train: 0.8434, loss_valid: 0.4711, accuracy_valid: 0.8328, 9.2460sec
batch_size: 2
epoch: 1, loss_train: 0.4142, accuracy_train: 0.8567, loss_valid: 0.4443, accuracy_valid: 0.8508, 4.7711sec
batch_size: 4
epoch: 1, loss_train: 0.4011, accuracy_train: 0.8607, loss_valid: 0.4367, accuracy_valid: 0.8635, 2.4349sec
batch_size: 8
epoch: 1, loss_train: 0.3951, accuracy_train: 0.8633, loss_valid: 0.4329, accuracy_valid: 0.8614, 1.2571sec
batch_size: 16
epoch: 1, loss_train: 0.3924, accuracy_train: 0.8636, loss_valid: 0.4317, accuracy_valid: 0.8593, 0.7718sec
batch_size: 32
epoch: 1, loss_train: 0.3912, accuracy_train: 0.8637, loss_valid: 0.4310, accuracy_valid: 0.8614, 0.4006sec
batch_size: 64
epoch: 1, loss_train: 0.3906, accuracy_train: 0.8643, loss_valid: 0.4307, accuracy_valid: 0.8624, 0.2672sec
batch_size: 128
epoch: 1, loss_train: 0.3904, accuracy_train: 0.8643, loss_valid: 0.4306, accuracy_valid: 0.8624, 0.1768sec
batch_size: 256
epoch: 1, loss_train: 0.3898, accuracy_train: 0.8643, loss_valid: 0.4305, accuracy_valid: 0.8624, 0.1403sec
batch_size: 512
epoch: 1, loss_train: 0.3901, accuracy_train: 0.8643, loss_valid: 0.4305, accuracy_valid: 0.8624, 0.1161sec
batch_size: 1024
epoch: 1, loss_train: 0.3953, accuracy_train: 0.8645, loss_valid: 0.4304, accuracy_valid: 0.8624, 0.1077sec
"""

