import joblib
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
from knock73 import CreateDataset
from knock74 import calc_accuracy
import time

class NN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.h1 = nn.Linear(input_size,hidden_size)
    self.h2 = nn.Linear(hidden_size,hidden_size)
    self.h3 = nn.Linear(hidden_size,hidden_size)
    self.output = nn.Linear(hidden_size, output_size)
    nn.init.normal_(self.h1.weight, 0.0, 1.0)
    nn.init.normal_(self.output.weight, 0.0, 1.0)

  def forward(self, x):
    x = F.relu(self.h1(x))
    x = F.relu(self.h2(x))
    x = F.relu(self.h3(x))
    x = self.output(x)
    return x

if __name__ == "__main__":
    trainf_np = joblib.load('train_feature.pkl')
    train_x = torch.from_numpy(trainf_np)
    trainl_np = joblib.load('train_label.pkl')
    train_y = torch.from_numpy(trainl_np)
    validf_np = joblib.load('valid_feature.pkl')
    valid_x = torch.from_numpy(validf_np)
    validl_np = joblib.load('valid_label.pkl')
    valid_y = torch.from_numpy(validl_np)
    testf_np = joblib.load('test_feature.pkl')
    test_x = torch.from_numpy(testf_np)
    testl_np = joblib.load('test_label.pkl')
    test_y = torch.from_numpy(testl_np)

    train_set = CreateDataset(train_x.float(),train_y)
    load_train = DataLoader(train_set, batch_size=64)
    valid_set = CreateDataset(valid_x.float(),valid_y)
    load_valid = DataLoader(valid_set)
    test_set = CreateDataset(test_x.float(),test_y)
    load_test = DataLoader(test_set)

    model = NN(300, 256, 4) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001)

    tr_loss_log = []
    val_loss_log = []
    tr_acc_log = []
    val_acc_log = []

    for epoch in range(50):
        model.train()
        train_loss = 0.0
        for i, (x,y) in enumerate(load_train):
            optimizer.zero_grad()
            prediction = model.forward(x)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()     

        model.eval()
        with torch.no_grad():
            x, y = next(iter(load_valid))
            pred = model.forward(x)
            val_loss = criterion(pred, y)

        train_loss = train_loss/i
        tr_loss_log.append(train_loss)
        val_loss_log.append(val_loss)
        
        tr_acc = calc_accuracy(model, load_train)
        val_acc = calc_accuracy(model, load_valid)
        tr_acc_log.append(tr_acc)
        val_acc_log.append(val_acc)   
        
        joblib.dump(model.state_dict(), 'checkpoint_' + str(epoch) + '.joblib')
        print(f'epoch: {epoch + 1}, train loss: {train_loss:.4f}, train accuracy: {tr_acc:.4f}, valid loss: {val_loss:.4f}, valid accuracy: {val_acc:.4f}') 
