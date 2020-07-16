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
dic_dir = "./dic"
pic_dir = "./picture/knock88_sub"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y): #x:(10284, each seq_len)
        self.x = x
        self.y = y

    def __len__(self):  # len(Dataset)で返す値を指定
        return len(self.y)

    def __getitem__(self, index):  # Dataset[index]で返す値を指定
        return self.x[index], self.y[index]


class LSTM(nn.Module):
    def __init__(self, vocab_size, padding_idx, dropout=0.5, emb_weights=None, embedding_dim=300, \
        hidden_dim=50, category_size=4, num_layers=1, bidirectional=False):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.category_size = category_size
        self.padding_idx = padding_idx
        self.num_layers = num_layers
        self.dropout1 = dropout
        self.dropout2 = 0.3
        self.num_directions = 2 if bidirectional==True else 1
        if emb_weights != None:
            self.word_embedding = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
        else: 
            self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        #self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, \
            dropout=self.dropout1, bidirectional=bidirectional)
        self.fc = nn.Linear(self.num_directions*hidden_dim, category_size)
        self.train_loss_list, self.valid_loss_list, self.train_acc_list, self.valid_acc_list, \
            self.test_acc_list, self.test_loss_list = [], [], [], [], [], []
    
    
    def forward(self, inputs):
        embeds = self.word_embedding(inputs)
        embeds = F.dropout(embeds, self.dropout2)
        # 1 * embedding_dim(300) * 単語数(len(sentence))のテンソル
        packed = nn.utils.rnn.pack_padded_sequence(embeds, self.lengths, batch_first=True, enforce_sorted=False)
        #戻り値：output, h_n
        #output: 全てのh_tが並べられているテンソル, h_n: 最後の出力層の隠れ状態ベクトル
        #input of shape (batch, seq_len, input_size)
        #output of shape (batch_size, seq_len, num_directions * hidden_size)
        #h_out of shape (num_layers * num_directions, batch, hidden_size)
        _, h = self.lstm(packed)
        h_out = h[0]
        #print(output)
        #print(len(output))
        h_out = self.transform(h_out)
        #output, output_lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        #print(output.shape)
        #print(output[0])       
        scores = self.fc(F.dropout(h_out, self.dropout2))
        #scores = self.fc(h_out.view(self.batch_size, self.hidden_dim))
        #print("scores", scores)
        probs = F.softmax(scores, dim=-1)
        #print(probs)
        return scores.squeeze()
    
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

    def Train(self, epoch_size=100, batch_size=80, learning_rate=0.001):
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        model = self
        dataset = MyDataset(self.train_ids, self.train_labels.to(device))
        #print(dataset[0:2])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        self.criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in tqdm(range(epoch_size)):
            for inputs, targets in dataloader:
                model.train()
                #バッチサイズに応じたパディングを施す
                inputs = self.padding(inputs).to(device)
                #勾配リセット
                optimizer.zero_grad()
                outputs = model(inputs.to(device))
                outputs = outputs.view(-1, self.category_size)
                loss = self.criterion(outputs, targets)
                #print(loss)
                #backpropagation
                loss.backward()
                optimizer.step()
            train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc = self.calc_loss_acc()
            self.train_loss_list.append(train_loss)
            self.valid_loss_list.append(valid_loss)
            self.test_loss_list.append(test_loss)
            self.train_acc_list.append(train_acc)
            self.valid_acc_list.append(valid_acc)
            self.test_acc_list.append(test_acc)

    def load_datasets(self):
        self.train_ids = torch.load(f"{data_dir}/train_id.list")
        self.valid_ids = torch.load(f"{data_dir}/valid_id.list")
        self.test_ids = torch.load(f"{data_dir}/test_id.list")
        self.train_labels = torch.load(f"{data_dir}/train_labels.tensor")
        self.valid_labels = torch.load(f"{data_dir}/valid_labels.tensor")
        self.test_labels = torch.load(f"{data_dir}/test_labels.tensor")

    def padding(self, inputs):
        if self.batch_size == 1:
            self.lengths = [len(inputs)]
            return torch.tensor(inputs)
        else:
            max_len = 0
            for sentence in inputs:
                if max_len < len(sentence):
                    max_len = len(sentence)
            new, self.lengths = [], []
            for sentence in inputs:
                self.lengths.append(len(sentence))
                sentence = sentence + [self.padding_idx] * (max_len - len(sentence))
                new.append(sentence)

            return torch.tensor(new)

    def calc_loss_acc(self):
        self.lengths = [1]
        #train
        train_pred = self.predict(self.train_ids)
        train_true = self.train_labels.to("cpu").detach().numpy()
        train_loss = self.cal_loss(self.train_ids, self.train_labels)
        train_acc = accuracy_score(train_true, train_pred).astype(np.float16)
        #valid
        valid_pred = self.predict(self.valid_ids)
        valid_true = self.valid_labels.to("cpu").detach().numpy()
        valid_loss = self.cal_loss(self.valid_ids, self.valid_labels)
        valid_acc = accuracy_score(valid_true, valid_pred).astype(np.float16)
        #test
        test_pred = self.predict(self.test_ids)
        test_true = self.test_labels.to("cpu").detach().numpy()
        test_loss = self.cal_loss(self.test_ids, self.test_labels)
        test_acc = accuracy_score(test_true, test_pred).astype(np.float16)

        return train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc

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
                loss = self.criterion(outputs, torch.tensor([label]).to(device)).to("cpu").detach().numpy().astype(np.float16)
                loss_l.append(loss)
        return sum(loss_l)/len(loss_l)

    #各サンプルの確率値から最も高い確率のインデックス（カテゴリ）をとる
    def select_class(self, probs):
        return np.argmax(probs.to("cpu").detach().numpy())
    
    def draw(self, lr=None, dropout=None, mode='show'):
        #エポック数のリスト
        x = list(range(self.epoch_size))
    
        fig = plt.figure(figsize=(12, 6))
        #lossのグラフ
        ax1 = fig.add_subplot(1,2,1)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        ax1.plot(x, self.train_loss_list, label='train')
        ax1.plot(x, self.valid_loss_list, label='valid')
        ax1.plot(x, self.test_loss_list, label='test')
        ax1.set_title('Loss')
        ax1.legend()
        #accuracyのグラフ
        ax2 = fig.add_subplot(1,2,2)
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('accuracy')
        ax2.plot(x, self.train_acc_list, label='train')
        ax2.plot(x, self.valid_acc_list, label='valid')
        ax2.plot(x, self.test_acc_list, label='test')
        ax2.set_title('Accuracy')
        ax2.legend()
    
        if lr!=None:
            fig.suptitle(f'learning rate : {lr}, dropout : {dropout}')

        if mode=='show':
            plt.show()
        else:
            plt.savefig(f'{pic_dir}/log_lr{lr}_dropout{dropout}.png')


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_embeds(vocab_size, embedding_dim, dic, ggl):
    set_seed()
    weights = np.zeros((vocab_size, embedding_dim))
    print(len(dic.values()))
    print(len(dic.keys()))
    cnt = 0
    c = 0
    for key, value in dic.items():
        if key in ggl:
            cnt += 1
            weights[value] = ggl[key]
        else:
            c += 1
            weights[value] = np.random.randn(1, embedding_dim) / np.sqrt(vocab_size)
    weights = torch.from_numpy(weights.astype(np.float32))
    print(cnt)
    print(c)
    return weights

def need_init_weights(vocab_size, padding_idx, dic, flg, embedding_dim=300):
    if flg:
        with open(f"{dic_dir}/google.pickle", mode='rb') as f:
            ggl = pickle.load(f)
        weights = init_embeds(vocab_size, embedding_dim, dic, ggl)
    else:
        weights = None
    return weights


class ParamSearch():
    def __init__(self, vocab_size, padding_idx, dic, weights_flg=False):
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.weights = need_init_weights(vocab_size, padding_idx, dic, weights_flg)
    
    def init_model(self, num_layers=1, bidirectional=False):
        self.num_layers = num_layers
        self.bidirectional = bidirectional
    
    def init_train(self, epoch_size=10, batch_size=1):
        self.epoch_size = epoch_size
        self.batch_size = batch_size
    
    def set_hyparam(self, param_dic):
        self.lr_param = param_dic["lr"]
        self.dropout_param = param_dic["dropout"]
    
    def search(self):
        max_valid_acc = 0
        for lr in self.lr_param:
            for dropout in self.dropout_param:
                set_seed()
                model = LSTM(self.vocab_size, padding_idx=self.padding_idx, dropout=dropout, emb_weights=self.weights, \
                    num_layers=self.num_layers, bidirectional=self.bidirectional)
                model.to(device)
                model.load_datasets()
                model.Train(epoch_size=self.epoch_size, batch_size=self.batch_size, learning_rate=lr)
                train_acc = model.train_acc_list[-1]
                valid_acc = model.valid_acc_list[-1]
                test_acc = model.test_acc_list[-1]
                print(f'learning rate : {lr:.5f}, dropout : {dropout:.1f}')
                print(f'train accuracy : {train_acc:.3f}, validation accuracy : {valid_acc:.3f}, test accuracy : {test_acc:.3f}')
                if max_valid_acc < valid_acc:
                    max_valid_acc = valid_acc
                    according_train_acc = train_acc
                    according_test_acc = test_acc
                    best_lr = lr
                    best_dropout = dropout
                    best_model = model
                else: pass
                model.draw(lr, dropout, mode='save')
        return best_model, [according_train_acc, max_valid_acc, according_test_acc], [best_lr, best_dropout]


def main():
    weights_flg = True
    embedding_dim = 300
    dic = torch.load(f"{dic_dir}/id_map.dic")
    vocab_size = len(set(dic.values())) + 1
    padding_idx = len(set(dic.values()))
    ps = ParamSearch(vocab_size, padding_idx, dic, weights_flg)
    ps.init_model(num_layers=2, bidirectional=True)
    ps.init_train(epoch_size=20, batch_size=64)
    ps.set_hyparam({
                    "lr":[0.00005, 0.00001],
                    "dropout":[0.1, 0.2, 0.3]
                    })
    best_model, accs, params = ps.search()
    print(f"train acc : {accs[0]}\nvalidation acc : {accs[1]}\ntest acc : {accs[2]}")
    print(f"lr : {params[0]:5f}\tdropout : {params[1]:1f}")
    #torch.save(best_model, "./model/best_model.model")
    #best_model.draw()


if __name__ == '__main__':
    main()

'''
100%|██████████████████████████████████████████████████████████████████████████████| 10/10 [11:30<00:00, 69.10s/it]
learning rate : 0.1000, dropout : 0.1
train accuracy : 0.816, validation accuracy : 0.779, test accuracy : 0.771
100%|██████████████████████████████████████████████████████████████████████████████| 10/10 [11:04<00:00, 66.48s/it]
learning rate : 0.1000, dropout : 0.2
train accuracy : 0.816, validation accuracy : 0.771, test accuracy : 0.775
100%|██████████████████████████████████████████████████████████████████████████████| 10/10 [10:57<00:00, 65.70s/it]
learning rate : 0.1000, dropout : 0.3
train accuracy : 0.809, validation accuracy : 0.765, test accuracy : 0.767
100%|██████████████████████████████████████████████████████████████████████████████| 10/10 [11:00<00:00, 66.01s/it]
learning rate : 0.1000, dropout : 0.4
train accuracy : 0.808, validation accuracy : 0.763, test accuracy : 0.769
100%|██████████████████████████████████████████████████████████████████████████████| 10/10 [11:31<00:00, 69.20s/it]
learning rate : 0.1000, dropout : 0.5
train accuracy : 0.802, validation accuracy : 0.766, test accuracy : 0.767
100%|██████████████████████████████████████████████████████████████████████████████| 10/10 [11:05<00:00, 66.59s/it]
learning rate : 0.0100, dropout : 0.1
train accuracy : 0.789, validation accuracy : 0.758, test accuracy : 0.766
100%|██████████████████████████████████████████████████████████████████████████████| 10/10 [11:02<00:00, 66.23s/it]
learning rate : 0.0100, dropout : 0.2
train accuracy : 0.786, validation accuracy : 0.753, test accuracy : 0.761
100%|██████████████████████████████████████████████████████████████████████████████| 10/10 [11:04<00:00, 66.42s/it]
learning rate : 0.0100, dropout : 0.3
train accuracy : 0.785, validation accuracy : 0.747, test accuracy : 0.765
100%|██████████████████████████████████████████████████████████████████████████████| 10/10 [11:02<00:00, 66.28s/it]
learning rate : 0.0100, dropout : 0.4
train accuracy : 0.780, validation accuracy : 0.757, test accuracy : 0.763
100%|██████████████████████████████████████████████████████████████████████████████| 10/10 [11:03<00:00, 66.33s/it]
learning rate : 0.0100, dropout : 0.5
train accuracy : 0.777, validation accuracy : 0.749, test accuracy : 0.760
100%|██████████████████████████████████████████████████████████████████████████████| 10/10 [11:08<00:00, 66.84s/it]
learning rate : 0.0010, dropout : 0.1
train accuracy : 0.684, validation accuracy : 0.680, test accuracy : 0.680
100%|██████████████████████████████████████████████████████████████████████████████| 10/10 [11:08<00:00, 66.85s/it]
learning rate : 0.0010, dropout : 0.2
train accuracy : 0.683, validation accuracy : 0.673, test accuracy : 0.684
100%|██████████████████████████████████████████████████████████████████████████████| 10/10 [11:08<00:00, 66.85s/it]
learning rate : 0.0010, dropout : 0.3
train accuracy : 0.679, validation accuracy : 0.674, test accuracy : 0.681
100%|██████████████████████████████████████████████████████████████████████████████| 10/10 [11:15<00:00, 67.55s/it]
learning rate : 0.0010, dropout : 0.4
train accuracy : 0.681, validation accuracy : 0.680, test accuracy : 0.677
100%|██████████████████████████████████████████████████████████████████████████████| 10/10 [11:13<00:00, 67.34s/it]
learning rate : 0.0010, dropout : 0.5
train accuracy : 0.680, validation accuracy : 0.679, test accuracy : 0.676
train acc : 0.81640625
validation acc : 0.779296875
test acc : 0.77099609375
lr : 0.100000   dropout : 0.100000
'''