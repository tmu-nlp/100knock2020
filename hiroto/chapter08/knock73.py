import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle

with open('./data/train_labels.pickle', mode='rb') as train_l\
        , open('./data/valid_labels.pickle', mode='rb') as valid_l\
        , open('./data/test_labels.pickle', mode='rb') as test_l\
        , open('./data/train_vectors.pickle', mode='rb') as train_v\
        , open('./data/valid_vectors.pickle', mode = 'rb') as valid_v\
        , open('./data/test_vectors.pickle', mode = 'rb') as test_v:
        train_labels = pickle.load(train_l)
        valid_labels = pickle.load(valid_l)
        test_labels = pickle.load(test_l)
        train_vectors = pickle.load(train_v)
        valid_vectors = pickle.load(valid_v)
        test_vectors = pickle.load(test_v)

class Net(nn.Module):
    def __init__(self, in_size, out_size):
        super(Net, self).__init__()
        self.fc = nn.Linear(in_size, out_size, bias=False)
    
    def forward(self, x):
        x = self.fc(x)
        x = F.softmax(x, dim=-1)
        return x

#一つの組が(vector, label)になっている
dataset = TensorDataset(train_vectors, train_labels)

#学習
def fit(model, epoch_size=100, batch_size=80):
    loss_list = []
    #ミニバッチするため，返り値はバッチサイズ分のデータセットが得られる
    dataloader = DataLoader(dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    for epoch in tqdm(range(epoch_size)):
        for inputs, targets in dataloader:
            #勾配リセット
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            #backpropagation
            loss.backward()
            optimizer.step()
        loss_list.append(loss)
    return model, loss_list

def draw(loss_list):
    x = list(range(len(loss_list)))
    plt.plot(x, loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def main():
    model = Net(300, 4)
    epoch_size = 100
    batch_size = 1
    model, loss_list = fit(model, epoch_size, batch_size)
    print(model.fc.weight.grad)
    draw(loss_list)
    #with open('./models/73model.pickle', mode='wb') as f:
       #pickle.dump(model, f)



if __name__ == '__main__':
    main()

'''
tensor([[ 1.0328e-06,  6.5007e-07, -8.6523e-07,  ..., -4.9794e-07,
         -2.9147e-07, -6.5142e-07],
        [ 7.1956e-06,  4.5293e-06, -6.0284e-06,  ..., -3.4693e-06,
         -2.0308e-06, -4.5387e-06],
        [-8.2311e-06, -5.1811e-06,  6.8959e-06,  ...,  3.9686e-06,
          2.3231e-06,  5.1918e-06],
        [ 3.3850e-09,  2.1307e-09, -2.8359e-09,  ..., -1.6321e-09,
         -9.5535e-10, -2.1351e-09]])
'''