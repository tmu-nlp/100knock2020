import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

from knock71 import SingleLayerPerceptron
from knock73 import CreateDataset, obtain_dataloader
from knock74 import calculate_accuracy
from knock75 import calculate_loss_and_accuracy, train_sgd_log, plot_loss_and_accuracy

def train_sgd_with_checkpoints(model, train_dataloader, dev_dataloader, model_name):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    num_epochs = 30
    train_log = []
    dev_log = []
    for epoch in range(num_epochs):
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

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'./checkpoints/{model_name}/checkpoint{epoch + 1}.pt')

        print(f'epoch: {epoch + 1}, loss_train: {train_loss:.4f}, accuracy_train: {train_acc:.4f}, loss_valid: {dev_loss:.4f}, accuracy_valid: {dev_acc:.4f}')
    return train_log, dev_log

if __name__ == '__main__':
    X_train = torch.load('./tensors/X_train')
    Y_train = torch.load('./tensors/Y_train')
    X_dev = torch.load('./tensors/X_dev')
    Y_dev = torch.load('./tensors/Y_dev')
    X_test = torch.load('./tensors/X_test')
    Y_test = torch.load('./tensors/Y_test')

    train_dataloader = obtain_dataloader(X_train, Y_train, batch_size=1, shuffle=True)
    dev_dataloader = obtain_dataloader(X_dev, Y_dev, batch_size=X_dev.shape[0], shuffle=False)

    model = SingleLayerPerceptron(300, 4)
    train_log, dev_log = train_sgd_with_checkpoints(model, train_dataloader, dev_dataloader, model_name='slp')


"""
epoch: 1, loss_train: 0.4519, accuracy_train: 0.8434, loss_valid: 0.4711, accuracy_valid: 0.8328
epoch: 2, loss_train: 0.3936, accuracy_train: 0.8649, loss_valid: 0.4328, accuracy_valid: 0.8593
epoch: 3, loss_train: 0.3663, accuracy_train: 0.8763, loss_valid: 0.4235, accuracy_valid: 0.8667
epoch: 4, loss_train: 0.3561, accuracy_train: 0.8782, loss_valid: 0.4243, accuracy_valid: 0.8677
epoch: 5, loss_train: 0.3390, accuracy_train: 0.8811, loss_valid: 0.4194, accuracy_valid: 0.8656
epoch: 6, loss_train: 0.3321, accuracy_train: 0.8869, loss_valid: 0.4195, accuracy_valid: 0.8656
epoch: 7, loss_train: 0.3277, accuracy_train: 0.8849, loss_valid: 0.4244, accuracy_valid: 0.8656
epoch: 8, loss_train: 0.3216, accuracy_train: 0.8913, loss_valid: 0.4260, accuracy_valid: 0.8646
epoch: 9, loss_train: 0.3176, accuracy_train: 0.8889, loss_valid: 0.4280, accuracy_valid: 0.8614
epoch: 10, loss_train: 0.3215, accuracy_train: 0.8893, loss_valid: 0.4340, accuracy_valid: 0.8593
epoch: 11, loss_train: 0.3152, accuracy_train: 0.8913, loss_valid: 0.4322, accuracy_valid: 0.8667
epoch: 12, loss_train: 0.3107, accuracy_train: 0.8923, loss_valid: 0.4314, accuracy_valid: 0.8603
epoch: 13, loss_train: 0.3087, accuracy_train: 0.8931, loss_valid: 0.4306, accuracy_valid: 0.8624
epoch: 14, loss_train: 0.3093, accuracy_train: 0.8947, loss_valid: 0.4338, accuracy_valid: 0.8667
epoch: 15, loss_train: 0.3058, accuracy_train: 0.8953, loss_valid: 0.4345, accuracy_valid: 0.8667
epoch: 16, loss_train: 0.3063, accuracy_train: 0.8953, loss_valid: 0.4360, accuracy_valid: 0.8646
epoch: 17, loss_train: 0.3066, accuracy_train: 0.8941, loss_valid: 0.4399, accuracy_valid: 0.8603
epoch: 18, loss_train: 0.3027, accuracy_train: 0.8958, loss_valid: 0.4397, accuracy_valid: 0.8603
epoch: 19, loss_train: 0.3023, accuracy_train: 0.8963, loss_valid: 0.4439, accuracy_valid: 0.8635
epoch: 20, loss_train: 0.3020, accuracy_train: 0.8963, loss_valid: 0.4440, accuracy_valid: 0.8593
epoch: 21, loss_train: 0.3027, accuracy_train: 0.8930, loss_valid: 0.4439, accuracy_valid: 0.8624
epoch: 22, loss_train: 0.3000, accuracy_train: 0.8984, loss_valid: 0.4444, accuracy_valid: 0.8656
epoch: 23, loss_train: 0.3046, accuracy_train: 0.8962, loss_valid: 0.4491, accuracy_valid: 0.8656
epoch: 24, loss_train: 0.3022, accuracy_train: 0.8961, loss_valid: 0.4497, accuracy_valid: 0.8603
epoch: 25, loss_train: 0.3002, accuracy_train: 0.8978, loss_valid: 0.4458, accuracy_valid: 0.8677
epoch: 26, loss_train: 0.3017, accuracy_train: 0.8975, loss_valid: 0.4533, accuracy_valid: 0.8614
epoch: 27, loss_train: 0.2986, accuracy_train: 0.8974, loss_valid: 0.4499, accuracy_valid: 0.8593
epoch: 28, loss_train: 0.2988, accuracy_train: 0.8984, loss_valid: 0.4535, accuracy_valid: 0.8646
epoch: 29, loss_train: 0.2982, accuracy_train: 0.8982, loss_valid: 0.4515, accuracy_valid: 0.8698
epoch: 30, loss_train: 0.2971, accuracy_train: 0.8987, loss_valid: 0.4505, accuracy_valid: 0.8688
"""

