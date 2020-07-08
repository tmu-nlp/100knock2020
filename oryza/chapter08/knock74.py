import joblib
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch
from knock71 import NN
from knock73 import CreateDataset

def calc_accuracy(model, loader):
    model.eval()
    total = 0
    correct = 0
    for x,y in loader:
        _, prediction = torch.max(model(x),1)
        total += len(y)
        correct += (prediction == y).sum().item()
    return correct/total

if __name__ == "__main__":
    trainf_np = joblib.load('train_feature.pkl')
    train_x = torch.from_numpy(trainf_np)
    trainl_np = joblib.load('train_label.pkl')
    train_y = torch.from_numpy(trainl_np)
    testf_np = joblib.load('test_feature.pkl')
    test_x = torch.from_numpy(testf_np)
    testl_np = joblib.load('test_label.pkl')
    test_y = torch.from_numpy(testl_np)

    train_set = CreateDataset(train_x.float(),train_y)
    load_train = DataLoader(train_set)
    test_set = CreateDataset(test_x.float(),test_y)
    load_test = DataLoader(test_set)

    model = NN(300, 256, 4) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001)

    for epoch in range(100):
        model.train()
        train_loss = 0.0
        for i, (x,y) in enumerate(load_train):
            optimizer.zero_grad()
            prediction = model.forward(x)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss = train_loss/i

        model.eval()
        with torch.no_grad():
            x, y = next(iter(load_test))
            pred = model.forward(x)
            loss_valid = criterion(pred, y)

    train_accuracy = calc_accuracy(model, load_train)
    test_accuracy = calc_accuracy(model, load_test)

    print('Train Accuracy: ' + str(round(train_accuracy,4)))
    print('Test Accuracy: ' + str(round(test_accuracy,4)))

'''
Train Accuracy: 0.5567
Test Accuracy: 0.5536
'''