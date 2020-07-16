# training_data_size ----> 10684 = 2^2 * 2671
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import pickle
data_dir = "./data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y): #x:(10284, each seq_len)
        self.x = x
        self.y = y

    def __len__(self):  # len(Dataset)で返す値を指定
        return len(self.y)

    def __getitem__(self, index):  # Dataset[index]で返す値を指定
        return self.x[index], self.y[index]


class CNN(nn.Module):
    def __init__(self, vocab_size, padding_idx, emb_weights=None, embedding_dim=300, \
        hidden_dim=50, category_size=4, num_layers=1, bidirectional=False, \
        kernel_size=3, stride=1, padding=1):
        super(CNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.category_size = category_size
        self.padding_idx = padding_idx
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional==True else 1
        if emb_weights != None:
            self.word_embedding = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
        else: 
            self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        #self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.conv = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.fc = nn.Linear(hidden_dim, category_size)
        self.train_loss_list, self.valid_loss_list, self.train_acc_list, \
            self.valid_acc_list = [], [], [], []
    
    
    def forward(self, inputs):
        embeds = self.word_embedding(inputs)
        #print(embeds.size())
        #print(embeds.view(embeds.shape[0], embeds.shape[2], embeds.shape[1]).size())
        p = F.relu(self.conv(embeds.view(embeds.shape[0], embeds.shape[2], embeds.shape[1])))
        #print(p.size())
        c = F.max_pool1d(p, p.shape[2])
        #print(c.size())
        scores = self.fc(c.view(c.shape[0], c.shape[2], c.shape[1]))
        #print(scores.size())
        #print(scores.squeeze(1))
        #print(scores.view(scores.shape[0], scores.shape[2]).shape)
        #probs = F.softmax(scores, dim=-1)
        return scores.view(scores.shape[0], scores.shape[2])

    def Train(self, epoch_size=100, batch_size=80, learning_rate=0.001):
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        model = self
        dataset = MyDataset(self.train_ids, self.train_labels.to(device))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        self.criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        for epoch in tqdm(range(epoch_size)):
            for inputs, targets in dataloader:
                model.train()
                #バッチサイズに応じたパディングを施す
                inputs = self.padding(inputs).to(device)
                #勾配リセット
                optimizer.zero_grad()
                outputs = model(inputs)
                outputs = outputs.view(-1, self.category_size)
                loss = self.criterion(outputs, targets)
                #print(loss)
                #backpropagation
                loss.backward()
                optimizer.step()
            train_loss, valid_loss, train_acc, valid_acc = self.calc_loss_acc()
            self.train_loss_list.append(train_loss)
            self.valid_loss_list.append(valid_loss)
            self.train_acc_list.append(train_acc)
            self.valid_acc_list.append(valid_acc)

    def load_datasets(self):
        self.train_ids = torch.load(f"{data_dir}/train_id.list")
        self.valid_ids = torch.load(f"{data_dir}/valid_id.list")
        self.test_ids = torch.load(f"{data_dir}/test_id.list")
        self.train_labels = torch.load(f"{data_dir}/train_labels.tensor")
        self.valid_labels = torch.load(f"{data_dir}/valid_labels.tensor")
        self.test_labels = torch.load(f"{data_dir}/test_labels.tensor")

    #多層とかBiの時とかのためにh_outを変形する
    def transform(self, h_out):
        #h_out of shape (num_layers * num_directions, batch, hidden_size)
        if h_out.shape[0] != 1:
            #h_out of shape (num_layers, num_directions, batch, hidden_size)
            h_out = h_out.view(self.num_layers, self.num_directions, h_out.shape[1], h_out.shape[2])
            if self.num_directions==1:
                #最後の層からの出力ベクトルを取得
                h_out = h_out[-1, :, :, :]
                #print(h_out.size())
            else:
                #最後の層からの出力ベクトルを取得
                #(num_layers, 2, batch. embed) ===> (2, batch, embed)
                h_out = h_out[-1, :, :, :]
                h_out = torch.cat((h_out[0], h_out[1]), dim=1)
                h_out = h_out.view(1, h_out.shape[0], h_out.shape[1])
                #print(h_out.size())
        else: pass
        #(1, batch_size, embedding_dim) ====> (batch_size, 1, embedding_dim)
        return h_out.view(-1, 1, h_out.shape[2])
    
    def collate_fn(self, batch):
        sentences, labels = list(zip(*batch))
        labels = torch.stack(labels)
        return sentences, labels

    def padding(self, inputs):
        if self.batch_size == 1:
            self.lengths = [len(inputs)]
            return torch.tensor(inputs)
        else:
            max_len = 0
            for sentence in inputs:
                if max_len < len(sentence):
                    max_len = len(sentence)
            new = []
            for sentence in inputs:
                sentence = self.adjust_padding(sentence, max_len)
                new.append(sentence)

            return torch.tensor(new)

    def adjust_padding(self, sentence, max_len):
        f = (max_len-len(sentence)) // 2
        b = (max_len-len(sentence)) - f
        sentence = [self.padding_idx]*f + sentence + [self.padding_idx]*b
        return sentence

    def calc_loss_acc(self):
        self.lengths = [1]
        #train
        train_pred = self.predict(self.train_ids)
        train_true = self.train_labels.detach().numpy()
        train_loss = self.cal_loss(self.train_ids, self.train_labels)
        train_acc = accuracy_score(train_true, train_pred).astype(np.float16)
        #valid
        valid_pred = self.predict(self.valid_ids)
        valid_true = self.valid_labels.detach().numpy()
        valid_loss = self.cal_loss(self.valid_ids, self.valid_labels)
        valid_acc = accuracy_score(valid_true, valid_pred).astype(np.float16)

        return train_loss, valid_loss, train_acc, valid_acc

    def predict(self, data):
        model = self
        model.eval()
        labels_pred = []
        with torch.no_grad():
            for sentence in data:
                sentence = torch.tensor(sentence).to(device)
                outputs = model(sentence.view(1, sentence.shape[0]))
                label = self.select_class(F.softmax(outputs, dim=-1))
                labels_pred.append(label)
        return labels_pred
    
    def cal_loss(self, data, labels):
        model = self
        model.eval()
        loss_l = []
        with torch.no_grad():
            for sentence, label in zip(data, labels):
                sentence = torch.tensor(sentence).to(device)
                outputs = model(sentence.view(1, sentence.shape[0]))
                outputs = outputs.view(-1, self.category_size)
                loss = self.criterion(outputs, torch.tensor([label])).detach().numpy().astype(np.float16)
                loss_l.append(loss)
        return sum(loss_l)/len(loss_l)

    #各サンプルの確率値から最も高い確率のインデックス（カテゴリ）をとる
    def select_class(self, probs):
        return np.argmax(probs.detach().numpy())
    
    def draw(self):
        #エポック数のリスト
        x = list(range(self.epoch_size))
    
        fig = plt.figure(figsize=(12, 6))
        #lossのグラフ
        ax1 = fig.add_subplot(1,2,1)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        ax1.plot(x, self.train_loss_list, label='train')
        ax1.plot(x, self.valid_loss_list, label='valid')
        ax1.set_title('Loss')
        ax1.legend()
        #accuracyのグラフ
        ax2 = fig.add_subplot(1,2,2)
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('accuracy')
        ax2.plot(x, self.train_acc_list, label='train')
        ax2.plot(x, self.valid_acc_list, label='valid')
        ax2.set_title('Accuracy')
        ax2.legend()
    
        plt.show()


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_embeds(vocab_size, embedding_dim, dic, ggl):
    set_seed()
    weights = np.zeros((vocab_size, embedding_dim))
    for key, value in dic.items():
        if key in ggl:
            weights[value] = ggl[key]
        else:
            weights[value] = np.random.randn(1, embedding_dim)
    weights = torch.from_numpy(weights.astype((np.float32)))
    return weights

def need_init_weights(flg, dic, vocab_size, padding_idx, embedding_dim=300):
    if flg:
        with open("./dic/google.pickle", mode='rb') as f:
            ggl = pickle.load(f)
        weights = init_embeds(vocab_size, embedding_dim, dic, ggl)
    else:
        weights = None
    return weights


def main():
    weights_flg = False
    embedding_dim = 300
    dic = torch.load("./dic/id_map.dic")
    vocab_size = len(set(dic.values())) + 1
    padding_idx = len(set(dic.values()))
    weights = need_init_weights(weights_flg, dic, vocab_size, padding_idx)
    set_seed()
    model = CNN(vocab_size, padding_idx=padding_idx, emb_weights=weights)
    model.to(device)
    model.load_datasets()
    model.Train(epoch_size=10, batch_size=1, learning_rate=0.001)
    model.draw()


if __name__ == '__main__':
    main()

'''
tensor([[[0.2483, 0.3293, 0.2090, 0.2133]]]
'''