import torch
from torch.utils.data import Dataset, DataLoader

from knock71 import SingleLayerPerceptron

class CreateDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y.reshape(-1).long()

    def __len__(self): 
        return len(self.y)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

def train_sgd(model):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    num_epochs = 10
    train_losses = []
    dev_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / i
        train_losses.append(train_loss)

        model.eval() 
        with torch.no_grad():
            inputs, labels = next(iter(dev_dataloader))
            outputs = model.forward(inputs)
            dev_loss = criterion(outputs, labels)
            dev_losses.append(dev_loss)

        print(f'epoch: {epoch + 1}, train_loss: {train_loss:.4f}, dev_loss: {dev_loss:.4f}')
    torch.save(model, 'SLP_sgd')
    return train_losses, dev_losses

def obtain_dataloader(X_data, Y_data, batch_size, shuffle):
    dataset = CreateDataset(X_data, Y_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=len(dev_dataset), shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    model = SingleLayerPerceptron(300, 4)
    train_losses, dev_losses = train_sgd(model)

"""
epoch: 1, train_loss: 0.6302, dev_loss: 0.4711
epoch: 2, train_loss: 0.4351, dev_loss: 0.4473
epoch: 3, train_loss: 0.3934, dev_loss: 0.4254
epoch: 4, train_loss: 0.3719, dev_loss: 0.4207
epoch: 5, train_loss: 0.3586, dev_loss: 0.4199
epoch: 6, train_loss: 0.3495, dev_loss: 0.4175
epoch: 7, train_loss: 0.3432, dev_loss: 0.4197
epoch: 8, train_loss: 0.3383, dev_loss: 0.4220
epoch: 9, train_loss: 0.3342, dev_loss: 0.4235
epoch: 10, train_loss: 0.3299, dev_loss: 0.4304
"""

