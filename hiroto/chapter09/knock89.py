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
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
data_dir = "./data"
pic_dir = "./picture"
MAX_LEN = -1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y): #x:(10284, each seq_len)
        global MAX_LEN
        MAX_LEN = max([len(tokenizer.tokenize(title)) for title in x])
        self.x, self.attention_masks = Tokenize(x, tokenizer)
        self.y = y

    def __len__(self):  # len(Dataset)で返す値を指定
        return len(self.y)

    def __getitem__(self, index):  # Dataset[index]で返す値を指定
        return self.x[index].to(device), self.attention_masks[index].to(device), self.y[index].to(device)

def Tokenize(titles, tokenizer):
    '''
    input_ids: list of token ids to be fed to a model
    token_type_ids: list of token type ids to be fed to a model
    attention_mask: list of indices specifying which tokens should be attended to by the model
    overflowing_tokens: list of overflowing tokens if a max length is specified.
    num_truncated_tokens: number of overflowing tokens a max_length is specified
    special_tokens_mask: if adding special tokens, this is a list of [0, 1], with 0 specifying special added tokens and 1 specifying sequence tokens.
    '''
    titles_ids, attention_masks = [], []
    for title in titles:
        dic = tokenizer.encode_plus(
                    title,
                    #文字列の先頭と最後にトークンを入れる
                    add_special_tokens = True,
                    max_length = MAX_LEN,
                    pad_to_max_length = True,
                    #どこがpaddingされているかわかる0,1のリスト
                    return_attention_mask = True,
                    #tensorで返す
                    return_tensors = 'pt'
                )
        titles_ids.append(dic["input_ids"])
        attention_masks.append(dic["attention_mask"])
    titles_ids = torch.cat(titles_ids)
    attention_masks = torch.cat(attention_masks)
    return titles_ids, attention_masks


class BERT(nn.Module):
    def __init__(self, category_size=4):
        super(BERT, self).__init__()
        self.category_size = category_size
        #config = BertConfig.from_pretrained("bert-base-uncased", num_labels = self.category_size, \
        #hidden_dropout_prob=0.3)
        self.bert = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels = self.category_size,
            output_attentions = False, # アテンションベクトルを出力するか
            output_hidden_states = False,
            hidden_dropout_prob=0.4
        )
        self.train_loss_list, self.valid_loss_list, self.train_acc_list, self.valid_acc_list, \
            self.test_acc_list, self.test_loss_list = [], [], [], [], [], []
    
    def forward(self, inputs, masks, labels, token_type_ids=None):
        loss, logits = self.bert(inputs, token_type_ids=token_type_ids, \
            attention_mask=masks, labels=labels)
        return loss, logits

    def Train(self, epoch_size=100, batch_size=80, learning_rate=0.001):
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        model = self
        dataset = MyDataset(self.train_title, self.train_labels)
        #print(dataset[0:2])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        for epoch in tqdm(range(epoch_size)):
            print(f"epoch : {epoch+1}")
            for inputs, masks, labels in dataloader:
                model.train()
                #勾配リセット
                optimizer.zero_grad()
                loss, logits = model(inputs, masks, labels)
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
        self.train_title = torch.load(f"{data_dir}/train_title.list")
        self.valid_title = torch.load(f"{data_dir}/valid_title.list")
        self.test_title = torch.load(f"{data_dir}/test_title.list")
        self.train_labels = torch.load(f"{data_dir}/train_labels.tensor")
        self.valid_labels = torch.load(f"{data_dir}/valid_labels.tensor")
        self.test_labels = torch.load(f"{data_dir}/test_labels.tensor")

    def calc_loss_acc(self):
        #train
        train_loss, train_pred = self.predict(self.train_title, self.train_labels)
        print("train loss : ", train_loss)
        train_true = self.train_labels.to("cpu").detach().numpy()
        train_acc = accuracy_score(train_true, train_pred).astype(np.float16)
        print("train acc : ", train_acc)
        #valid
        valid_loss, valid_pred = self.predict(self.valid_title, self.valid_labels)
        print("valid loss : ", valid_loss)
        valid_true = self.valid_labels.to("cpu").detach().numpy()
        valid_acc = accuracy_score(valid_true, valid_pred).astype(np.float16)
        print("valid acc : ", valid_acc)
        #test
        test_loss, test_pred = self.predict(self.test_title, self.test_labels)
        print("test loss : ", test_loss)
        test_true = self.test_labels.to("cpu").detach().numpy()
        test_acc = accuracy_score(test_true, test_pred).astype(np.float16)
        print("test acc : ", test_acc)
        return train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc

    def predict(self, titles, labels):
        model = self
        titles, masks = Tokenize(titles, tokenizer)
        labels_pred = []
        losses = []
        running_loss = 0
        cnt = 0
        model.eval()
        with torch.no_grad():
            for title, mask, label in zip(titles, masks, labels):
                cnt += 1
                loss, logits = model(title.view(1, MAX_LEN).to(device), \
                mask.view(1, MAX_LEN).to(device), label.to(device))
                running_loss += loss.to("cpu").detach().numpy()
                labels_pred.append(self.select_class(logits))
        return running_loss/cnt, labels_pred

    #各サンプルの確率値から最も高い確率のインデックス（カテゴリ）をとる
    def select_class(self, logits):
        return np.argmax(logits.to("cpu").detach().numpy())
    
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

        plt.savefig("./picture/bert_dropout0.4.png")

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    set_seed()
    model = BERT()
    model.to(device)
    model.load_datasets()
    model.Train(epoch_size=15, batch_size=32, learning_rate=2e-5)
    model.draw()
    #best_model.draw()

if __name__ == '__main__':
    main()
