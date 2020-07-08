import os
import time
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle
import pickle
#GPUを指定
device = torch.device("cuda")

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
dataset = TensorDataset(train_vectors.to(device), train_labels.to(device))
'''
#ミニバッチするため，返り値はバッチサイズ分のデータセットが得られる
dataloader = DataLoader(dataset, batch_size=10)
'''
#学習
def fit(model, epoch_size=100, batch_size=80):
    #ミニバッチするため，返り値はバッチサイズ分のデータセットが得られる
    dataloader = DataLoader(dataset, batch_size=batch_size)
    train_loss_list, valid_loss_list, train_acc_list, valid_acc_list = [], [], [], []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    elapsed_list = []
    for epoch in tqdm(range(epoch_size)):
        #計測開始
        start = time.time()
        for inputs, targets in dataloader:
            #勾配リセット
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            #backpropagation
            loss.backward()
            optimizer.step()
        #測定終了
        elapsed = time.time() - start
        elapsed_list.append(elapsed)
        train_loss, valid_loss, train_acc, valid_acc = calc_loss_acc(model, criterion)
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)
        torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'train_loss':train_loss,
            'valid_loss':valid_loss
        }, f'./77_checkpoints/batch_size{batch_size}/checkpoint{epoch+1}.pt')
    #経過時間の平均を算出
    mean = sum(elapsed_list)/len(elapsed_list)
    print(f"train_loss_first:{train_loss_list[0]}")
    print(f'batch_size:{batch_size:3d}, elapsed time:{mean} sec')
    return [train_loss_list, valid_loss_list], [train_acc_list, valid_acc_list]

def calc_loss_acc(model, criterion):
    #train
    outputs = model(train_vectors)
    train_pred = select_class(outputs)
    train_true = train_labels.numpy().copy()
    train_loss = criterion(outputs, train_labels)
    train_acc = accuracy_score(train_true, train_pred)
    #valid
    outputs = model(valid_vectors)
    valid_pred = select_class(outputs)
    valid_true = valid_labels.numpy().copy()
    valid_loss = criterion(outputs, valid_labels)
    valid_acc = accuracy_score(valid_true, valid_pred)

    return train_loss, valid_loss, train_acc, valid_acc

#各サンプルの確率値から最も高い確率のインデックス（カテゴリ）をとって，labelsにappend
def select_class(probs_list):
    labels = []
    for probs in probs_list:
        label = np.argmax(probs.detach().numpy().copy())
        labels.append(label)
    return labels


def draw(loss_lists, acc_lists, batch_size):
    #エポック数のリスト
    x = list(range(len(loss_lists[0])))
    
    fig = plt.figure(figsize=(12, 6))
    #lossのグラフ
    ax1 = fig.add_subplot(1,2,1)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.plot(x, loss_lists[0], label='train')
    ax1.plot(x, loss_lists[1], label='valid')
    ax1.set_title('Loss')
    ax1.legend()
    #accuracyのグラフ
    ax2 = fig.add_subplot(1,2,2)
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')
    ax2.plot(x, acc_lists[0], label='train')
    ax2.plot(x, acc_lists[1], label='valid')
    ax2.set_title('Accuracy')
    ax2.legend()
    
    plt.savefig(f"./77_checkpoints/picture/loss_acc_BatchSize{batch_size}.png")


def main():
    epoch_size = 100
    batch_size_list = [2**i for i in range(10)]
    for batch_size in batch_size_list:
        print(f"#############batch_size{batch_size}#######################")
        fname = f"./77_checkpoints/batch_size{batch_size}"
        if os.path.isdir(fname): pass
        else: os.mkdir(fname)
        model = Net(300, 4).to(device)
        loss_lists, acc_lists = fit(model, epoch_size, batch_size)
        draw(loss_lists, acc_lists, batch_size)

if __name__ == '__main__':
    main()

'''
#############batch_size1#######################
train_loss_first:1.3774560689926147
batch_size:  1, elapsed time:8.278033483028413 sec
#############batch_size2#######################
train_loss_first:1.3850730657577515
batch_size:  2, elapsed time:4.602236063480377 sec
#############batch_size4#######################
train_loss_first:1.3821003437042236
batch_size:  4, elapsed time:2.3698193550109865 sec
#############batch_size8#######################
train_loss_first:1.378899335861206
batch_size:  8, elapsed time:1.1785529160499573 sec
#############batch_size16#######################
train_loss_first:1.3849132061004639
batch_size: 16, elapsed time:0.6430390405654908 sec
#############batch_size32#######################
train_loss_first:1.387450933456421
batch_size: 32, elapsed time:0.3507828998565674 sec
#############batch_size64#######################
train_loss_first:1.3866002559661865
batch_size: 64, elapsed time:0.2258523416519165 sec
#############batch_size128#######################
train_loss_first:1.385908603668213
batch_size:128, elapsed time:0.16308969259262085 sec
#############batch_size256#######################
train_loss_first:1.3869508504867554
batch_size:256, elapsed time:0.12377050161361694 sec
#############batch_size512#######################
train_loss_first:1.3844016790390015
batch_size:512, elapsed time:0.10136705875396729 sec
'''