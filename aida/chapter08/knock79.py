import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt

from knock71 import SingleLayerPerceptron
from knock73 import CreateDataset, obtain_dataloader
from knock74 import calculate_accuracy
from knock75 import calculate_loss_and_accuracy, train_sgd_log, plot_loss_and_accuracy
from knock76 import train_sgd_with_checkpoints
from knock77 import train_sgd_minibatch
from knock78 import calculate_loss_and_accuracy_with_gpu, train_sgd_with_gpu

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_size, mid_size, output_size, mid_layers):
        super().__init__()
        self.mid_layers = mid_layers
        self.fc = torch.nn.Linear(input_size, mid_size)
        self.fc_mid = torch.nn.Linear(mid_size, mid_size)
        self.fc_out = torch.nn.Linear(mid_size, output_size) 
        self.bn = torch.nn.BatchNorm1d(mid_size)

    def forward(self, x):
        x = F.relu(self.fc(x))
        for _ in range(self.mid_layers):
        x = F.relu(self.bn(self.fc_mid(x)))
        x = F.relu(self.fc_out(x))

        return x

def calculate_accuracy_with_gpu(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()

    return correct / total

if __name__ == '__main__':
    X_train = torch.load('./tensors/X_train')
    Y_train = torch.load('./tensors/Y_train')
    X_dev = torch.load('./tensors/X_dev')
    Y_dev = torch.load('./tensors/Y_dev')
    X_test = torch.load('./tensors/X_test')
    Y_test = torch.load('./tensors/Y_test')

    train_dataset = CreateDataset(X_train, Y_train)
    dev_dataset = CreateDataset(X_dev, Y_dev)
    test_dataset = CreateDataset(X_test, Y_test)

    model = MultiLayerPerceptron(300, 200, 4, 1)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    device = torch.device('cuda')
    log = train_sgd_with_gpu(train_dataset, dev_dataset, batch_size=128, model, criterion, optimizer, 1, device=device, model_name='mlp')

    train_log = log['train']
    dev_log = log['dev']
    plot_loss_and_accuracy(train_log, dev_log, file_name='MLP_log.png')

    acc_train = calculate_accuracy(model, train_dataloader, device)
    acc_test = calculate_accuracy(model, test_dataloader, device)
    print(f'Accuracy (train): {acc_train:.3f}')
    print(f'Accuracy (test): {acc_test:.3f}')

"""
Accuracy (train): 0.999
Accuracy (test): 0.878

# SingleLayerPerceptron
Accuracy (train): 0.891
Accuracy (test): 0.858
"""

