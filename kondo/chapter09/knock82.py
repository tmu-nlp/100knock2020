"""
82. 確率的勾配降下法による学習Permalink
確率的勾配降下法（SGD: Stochastic Gradient Descent）を用いて，問題81で構築したモデルを学習せよ．
訓練データ上の損失と正解率，評価データ上の損失と正解率を表示しながらモデルを学習し，適当な基準（例えば10エポックなど）で終了させよ．
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from matplotlib import pyplot as plt
import time
from tqdm import tqdm

from knock80 import translate_to_id
from knock81 import get_X_Y, CreateData

train_path = "../chapter06/train.csv"
valid_path = "../chapter06/valid.csv"
test_path = "../chapter06/test.csv"

_, V = translate_to_id("")

VOCAB_SIZE = V+1    #vocab + padding
EMB_SIZE = 300
PADDING_IDX = V     #padding
OUTPUT_SIZE = 4
HIDDEN_SIZE = 50

class RNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, hidden_size):
        torch.manual_seed(7)
        super().__init__()
        self.hidden_size = hidden_size
        #単語IDを与えるとone-hotベクトルに変換
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        #emb.size() = (batch_size, seq_len, emb_size)なのでそれに合わせるためにbatch_size = True(元は(seq_len, batch_size, emb_size))
        self.rnn = nn.RNN(emb_size, hidden_size, nonlinearity="tanh", batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def sinf(self, x):
        print("asg")

    def forward(self, x):
        #h0の初期化
        self.batch_size = x.size()[0]
        hidden = self.init_hidden()
        emb = self.emb(x)
        #out は時系列に対応する出力
        out, hidden = self.rnn(emb, hidden.to('cuda'))

        out = self.fc(out[:, -1])
        return out

    def init_hidden(self):
        #batch_size*hidden_sizeを1つ
        hidden = torch.zeros(1, self.batch_size, self.hidden_size)
        return hidden

def train_model(tr_data, va_data, model, criterion, optimizer, epochs=10, batch_size=1, collate_fn=None, device=None):

    model.to(device)

    #collate_fnはミニバッチ化する時にサイズを揃える時に使える。基本はtensorを返すだけ
    tr_l_data = DataLoader(tr_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    va_l_data = DataLoader(va_data, batch_size=1, shuffle=False)

    #スケジューラの設定cosine関数に従って学習率をepoch数で初期値からeta_minまで小さくする
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-5, last_epoch=-1)

    total_time = 0
    train_log = []
    valid_log = []
    for i in tqdm(range(epochs)):
        s_time = time.time()

        model.train()
        total_loss=0.0

        for j, X in enumerate(tr_l_data):
            optimizer.zero_grad()          #勾配の初期化

            #デバイスの指定
            inputs = X["id_text"].to(device)
            labels = X["labels"].to(device)

            outputs = model.forward(inputs)  #順伝播
            loss = criterion(outputs, labels)
            loss.backward()                     #逆伝播
            optimizer.step()               #重み更新

            total_loss += loss.item()
        train_loss = total_loss/(j+1)              #バッチ単位のロス

        model.eval()

        train_loss, train_acc = calculate_loss_acc(tr_data, model, criterion, device)
        valid_loss, valid_acc = calculate_loss_acc(va_data, model, criterion, device)
        train_log.append([train_loss, train_acc])
        valid_log.append([valid_loss, valid_acc])

        torch.save({"epoch": i+1, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, f"checkpoinit{i+1}.pt")

        e_time = time.time()

        print(f"epoch: {i+1}, train loss: {train_loss:.5f}, train accuracy: {train_acc}, valid loss: {valid_loss}, valid accuracy: {valid_acc}, time: {(e_time-s_time):.5f}sec")

        total_time += e_time-s_time
        print(f"1エポックあたり{total_time/(i+1)}s\n")

        #検証データのロスが3エポック連続で低下しなかったら学習終了
        if i > 2 and valid_log[i-3][0] <= valid_log[i-2][0] <= valid_log[i-1][0] <= valid_log[i][0]:
            break

        scheduler.step()

    return {"train": train_log, "valid": valid_log}

def calculate_loss_acc(data, model, criterion, device=None):
    l_data = DataLoader(data, batch_size=1, shuffle=False)
    loss = 0
    total = 0
    cor = 0
    with torch.no_grad():
        for X in l_data:
            #デバイスの指定
            inputs = X["id_text"].to(device)
            labels = X["labels"].to(device)

            outputs = model(inputs)

            loss += criterion(outputs, labels)

            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            cor += (pred == labels).sum().item()
    return loss/len(l_data), cor/total

def plot(log):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(np.array(log["train"]).T[0], label="train")
    ax[0].plot(np.array(log["valid"]).T[0], label="valid")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].legend()
    ax[1].plot(np.array(log["train"]).T[1], label="train")
    ax[1].plot(np.array(log["valid"]).T[1], label="valid")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].legend()
    plt.show()

if __name__ == "__main__":
    train_X, train_Y = get_X_Y(train_path)
    valid_X, valid_Y = get_X_Y(valid_path)
    test_X, test_Y = get_X_Y(test_path)
    train_dataset = CreateData(train_X, train_Y, translate_to_id)
    valid_dataset = CreateData(valid_X, valid_Y, translate_to_id)
    test_dataset = CreateData(test_X, test_Y, translate_to_id)

    device = torch.device('cuda')

    model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    log = train_model(train_dataset, valid_dataset, model, criterion, optimizer, epochs=10, batch_size=1, device=device)

    plot(log)

    _, train_acc = calculate_loss_acc(train_dataset, model, criterion, device)
    _, test_acc = calculate_loss_acc(test_dataset,model, criterion, device)
    print(f"train accuracy: {train_acc}")
    print(f"test accuracy: {test_acc}")

"""
＊1エポックあたりの時間間違い
 10%|█         | 1/10 [04:34<41:12, 274.76s/it]epoch: 1, train loss: 1.08629, train accuracy: 0.5453949831523773, valid loss: 1.1073004007339478, valid accuracy: 0.5127245508982036, time: 274.76351sec
1エポックあたり27.476351070404053s

 20%|██        | 2/10 [09:10<36:41, 275.17s/it]epoch: 2, train loss: 0.98187, train accuracy: 0.6175589666791463, valid loss: 1.0418548583984375, valid accuracy: 0.5823353293413174, time: 276.11305sec
1エポックあたり27.61130545139313s

 30%|███       | 3/10 [13:47<32:09, 275.67s/it]epoch: 3, train loss: 0.81607, train accuracy: 0.7092849120179708, valid loss: 0.900628387928009, valid accuracy: 0.6714071856287425, time: 276.84030sec
1エポックあたり27.68402979373932s

 40%|████      | 4/10 [18:24<27:36, 276.10s/it]epoch: 4, train loss: 0.68431, train accuracy: 0.7631037064769749, valid loss: 0.7822988033294678, valid accuracy: 0.7380239520958084, time: 277.07025sec
1エポックあたり27.707025146484376s

 50%|█████     | 5/10 [23:03<23:04, 276.88s/it]epoch: 5, train loss: 0.59632, train accuracy: 0.7909022837888431, valid loss: 0.7489058375358582, valid accuracy: 0.7544910179640718, time: 278.69049sec
1エポックあたり27.86904890537262s

 60%|██████    | 6/10 [27:42<18:30, 277.51s/it]epoch: 6, train loss: 0.53462, train accuracy: 0.8105578435043055, valid loss: 0.718532145023346, valid accuracy: 0.7619760479041916, time: 278.99977sec
1エポックあたり27.899976682662963s

 70%|███████   | 7/10 [32:22<13:54, 278.14s/it]epoch: 7, train loss: 0.48432, train accuracy: 0.8258143017596405, valid loss: 0.6897792816162109, valid accuracy: 0.7739520958083832, time: 279.59567sec
1エポックあたり27.959567141532897s

 80%|████████  | 8/10 [37:02<09:17, 278.97s/it]epoch: 8, train loss: 0.45017, train accuracy: 0.840134780980906, valid loss: 0.6833752989768982, valid accuracy: 0.7739520958083832, time: 280.89106sec
1エポックあたり28.089105677604675s

 90%|█████████ | 9/10 [41:42<04:39, 279.17s/it]epoch: 9, train loss: 0.42836, train accuracy: 0.8487457880943466, valid loss: 0.6847438216209412, valid accuracy: 0.7791916167664671, time: 279.64984sec
1エポックあたり27.964984464645386s

100%|██████████| 10/10 [46:23<00:00, 278.34s/it]epoch: 10, train loss: 0.42227, train accuracy: 0.8481842006739049, valid loss: 0.686535120010376, valid accuracy: 0.781437125748503, time: 280.79659sec
1エポックあたり28.079659199714662s
"""








































































"""
82. 確率的勾配降下法による学習Permalink
確率的勾配降下法（SGD: Stochastic Gradient Descent）を用いて，問題81で構築したモデルを学習せよ．
訓練データ上の損失と正解率，評価データ上の損失と正解率を表示しながらモデルを学習し，適当な基準（例えば10エポックなど）で終了させよ．
"""
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from matplotlib import pyplot as plt
import time
from tqdm import tqdm

from knock80 import translate_to_id
from knock81 import RNN, get_X_Y, CreateData

train_path = "../chapter06/train.csv"
valid_path = "../chapter06/valid.csv"
test_path = "../chapter06/test.csv"

_, V = translate_to_id("")

VOCAB_SIZE = V
EMB_SIZE = 300
PADDING_IDX = 0
OUTPUT_SIZE = 4
HIDDEN_SIZE = 50

class SGC():
    def __init__(self, model):
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        #オプティマイザ
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)

    def fit(self, tr_data, va_data, epochs=10, batch_size=1, collate_fn=None, device=None):

        self.model.to(device)
        #collate_fnはミニバッチ化する時にサイズを揃える時に使える。基本はtensorを返すだけ
        tr_l_data = DataLoader(tr_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        va_l_data = DataLoader(va_data, batch_size=1, shuffle=False)

        #スケジューラの設定cosine関数に従って学習率をepoch数で初期値からeta_minまで小さくする
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs, eta_min=1e-5, last_epoch=-1)

        train_log = []
        valid_log = []
        for i in tqdm(range(epochs)):
            total_time = 0
            s_time = time.time()

            self.model.train()
            total_loss=0.0

            for j, X in enumerate(tr_l_data):
                self.optimizer.zero_grad()          #勾配の初期化

                #デバイスの指定
                inputs = X["id_text"].to(device)
                labels = X["labels"].to(device)

                outputs = self.model.forward(inputs)  #順伝播
                loss = self.criterion(outputs, labels)
                loss.backward()                     #逆伝播
                self.optimizer.step()               #重み更新

                total_loss += loss.item()
            train_loss = total_loss/(j+1)              #バッチ単位のロス

            self.model.eval()

            train_loss, train_acc = self.calculate_loss_acc(tr_data, device)
            valid_loss, valid_acc = self.calculate_loss_acc(va_data, device)
            train_log.append([train_loss, train_acc])
            valid_log.append([valid_loss, valid_acc])

            torch.save({"epoch": i+1, "model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict()}, f"checkpoinit{i+1}.pt")

            e_time = time.time()

            print(f"epoch: {i+1}, train loss: {train_loss:.5f}, train accuracy: {train_acc}, valid loss: {valid_loss}, valid accuracy: {valid_acc}, time: {(e_time-s_time):.5f}sec")

            total_time += e_time-s_time
            print(f"1エポックあたり{total_time/epochs}s\n")

            #検証データのロスが3エポック連続で低下しなかったら学習終了
            if i > 2 and valid_log[i-3][0] <= valid_log[i-2][0] <= valid_log[i-1][0] <= valid_log[i][0]:
                break

            scheduler.step()

        return {"train": train_log, "valid": valid_log}

    def calculate_loss_acc(self, data, device=None):
        l_data = DataLoader(data, batch_size=1, shuffle=False)
        loss = 0
        total = 0
        cor = 0
        with torch.no_grad():
            for X in l_data:
                #デバイスの指定
                inputs = X["id_text"].to(device)
                labels = X["labels"].to(device)

                outputs = self.model(inputs)

                loss += self.criterion(outputs, labels)

                pred = torch.argmax(outputs, dim=-1)
                total += len(inputs)
                cor += (pred == labels).sum().item()
        return loss/len(l_data), cor/total

    def plot(self, train_log, valid_log, flag):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].plot(np.array(train_log).T[0], label="train")
        ax[0].plot(np.array(valid_log).T[0], label="valid")
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("loss")
        ax[0].legend()
        ax[1].plot(np.array(train_log).T[1], label="train")
        ax[1].plot(np.array(valid_log).T[1], label="valid")
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("accuracy")
        ax[1].legend()
        if flag == 1:
            plt.show()
        else:
            plt.pause(.01)

if __name__ == "__main__":
    train_X, train_Y = get_X_Y(train_path)
    valid_X, valid_Y = get_X_Y(valid_path)
    test_X, test_Y = get_X_Y(test_path)
    train_dataset = CreateData(train_X, train_Y, translate_to_id)
    valid_dataset = CreateData(valid_X, valid_Y, translate_to_id)
    test_dataset = CreateData(test_X, test_Y, translate_to_id)

    device = torch.device('cuda')

    sgc = SGC(RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE))

    log = sgc.fit(train_dataset, valid_dataset, epochs=10, batch_size=1, device=device)
"""