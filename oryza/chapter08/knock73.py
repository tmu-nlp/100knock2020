import joblib
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch
from knock71 import NN

class CreateDataset(Dataset):
  def __init__(self, X, y): 
    self.X = X
    self.y = y

  def __len__(self): 
    return len(self.y)

  def __getitem__(self, idx): 
    return [self.X[idx], self.y[idx]]

if __name__ == "__main__":
    trainf_np = joblib.load('train_feature.pkl')
    train_x = torch.from_numpy(trainf_np)
    trainl_np = joblib.load('train_label.pkl')
    train_y = torch.from_numpy(trainl_np)

    train_set = CreateDataset(train_x.float(),train_y)
    load_train = DataLoader(train_set)

    model = NN(300, 256, 4) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(100):
        model.train()
        loss_log = 0.0
        for i, (x,y) in enumerate(load_train):
            optimizer.zero_grad()
            prediction = model.forward(x)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()
            loss_log += loss.item()
        
        loss_log = loss_log/i

        model.eval()
        with torch.no_grad():
            x, y = next(iter(load_train))
            pred = model.forward(x)
            loss_valid = criterion(pred, y)

    print(model.state_dict()['output.weight'])

'''
tensor([[-0.7660, -0.1651,  0.9225,  ..., -0.4826, -0.7915,  0.2288],
        [-0.7972, -0.2686,  0.8378,  ..., -0.3485, -0.7809,  0.3550],
        [-0.8559, -0.2597,  0.9871,  ..., -0.3793, -0.8799,  0.2466],
        [-0.7446, -0.2319,  1.0148,  ..., -0.3352, -0.7956,  0.3018]])
'''