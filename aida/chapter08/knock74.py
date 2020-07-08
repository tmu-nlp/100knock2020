import torch
from torch.utils.data import Dataset, DataLoader

from knock71 import SingleLayerPerceptron
from knock73 import CreateDataset, obtain_dataloader

def calculate_accuracy(model, loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
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

    train_dataloader = obtain_dataloader(X_train, Y_train, batch_size=1, shuffle=True)
    test_dataloader = obtain_dataloader(X_test, Y_test, batch_size=X_test.shape[0], shuffle=False)

    model = torch.load('./models/SLP_sgd')

    train_acc = calculate_accuracy(model, train_dataloader)
    test_acc = calculate_accuracy(model, test_dataloader)
    print(f'Accuracy (train): {train_acc:.3f}')
    print(f'Accuracy (test): {test_acc:.3f}')

"""
Accuracy (train): 0.891
Accuracy (test): 0.858
"""

