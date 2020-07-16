"""
83. ミニバッチ化・GPU上での学習Permalink
問題82のコードを改変し，B事例ごとに損失・勾配を計算して学習を行えるようにせよ（Bの値は適当に選べ）．また，GPU上で学習を実行せよ．
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

VOCAB_SIZE = V+1
EMB_SIZE = 300
PADDING_IDX = V
OUTPUT_SIZE = 4
HIDDEN_SIZE = 50

class RNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, hidden_size):
        torch.manual_seed(7)
        super().__init__()
        self.hidden_size = hidden_size
        #単語IDを与えるとone-hotベクトルに変換　paddingで作られた本来存在しない要素に対応するベクトルは0にする
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
        #hiddenだけ何故かGPUに送られなくで怒られたのでここで送る
        out, hidden = self.rnn(emb, hidden.to('cuda'))

        out = self.fc(out[:, -1])
        return out

    def init_hidden(self):
        #batch_size*hidden_sizeを1つ
        hidden = torch.zeros(1, self.batch_size, self.hidden_size)
        return hidden

class Padsequence():
    #バッチ内での単語長は同じでなければならないので長いやつに合わせてパディング
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx

    def __call__(self, batch):
        #バッチ内で単語の長い順にソート
        sorted_batch = sorted(batch, key=lambda x: x["id_text"].shape[0], reverse=True)
        sequences = [x["id_text"] for x in sorted_batch]
        #
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.padding_idx)
        labels = torch.LongTensor([x["labels"] for x in sorted_batch])

        return {"id_text": sequences_padded, "labels": labels}

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

def result(train_dataset, test_dataset, model, criterion, device):
    _, train_acc = calculate_loss_acc(train_dataset, model, criterion, device)
    _, test_acc = calculate_loss_acc(test_dataset, model, criterion, device)
    print(f"train accuracy: {train_acc}")
    print(f"test accuracy: {test_acc}")

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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    log = train_model(train_dataset, valid_dataset, model, criterion, optimizer, epochs=10, batch_size=128, collate_fn=Padsequence(PADDING_IDX), device=device)

    plot(log)

    result(train_dataset, test_dataset, model, criterion, device)

"""
 0%|          | 0/10 [00:00<?, ?it/s]
 10%|█         | 1/10 [04:06<37:02, 246.93s/it]epoch: 1, train loss: 1.27404, train accuracy: 0.3672781729689255, valid loss: 1.2609988451004028, valid accuracy: 0.3555389221556886, time: 246.93180sec
1エポックあたり246.93179774284363s


 20%|██        | 2/10 [08:14<32:57, 247.16s/it]epoch: 2, train loss: 1.25916, train accuracy: 0.37411081991763384, valid loss: 1.2474305629730225, valid accuracy: 0.3630239520958084, time: 247.66912sec
1エポックあたり247.30045890808105s


 30%|███       | 3/10 [12:25<28:57, 248.29s/it]epoch: 3, train loss: 1.25536, train accuracy: 0.3822538375140397, valid loss: 1.244576096534729, valid accuracy: 0.36601796407185627, time: 250.91403sec
1エポックあたり248.50498135884604s


 40%|████      | 4/10 [16:36<24:54, 249.14s/it]epoch: 4, train loss: 1.24939, train accuracy: 0.38946087607637586, valid loss: 1.2398744821548462, valid accuracy: 0.3712574850299401, time: 251.10843sec
1エポックあたり249.1558443903923s


 50%|█████     | 5/10 [20:46<20:46, 249.32s/it]epoch: 5, train loss: 1.24745, train accuracy: 0.3985398727068514, valid loss: 1.2395168542861938, valid accuracy: 0.3847305389221557, time: 249.74571sec
1エポックあたり249.27381772994994s


 60%|██████    | 6/10 [24:59<16:42, 250.54s/it]epoch: 6, train loss: 1.24434, train accuracy: 0.40527892175215274, valid loss: 1.2381459474563599, valid accuracy: 0.39146706586826346, time: 253.37265sec
1エポックあたり249.95695638656616s


 70%|███████   | 7/10 [29:14<12:35, 251.69s/it]epoch: 7, train loss: 1.24333, train accuracy: 0.41164357918382627, valid loss: 1.23848557472229, valid accuracy: 0.39895209580838326, time: 254.36135sec
1エポックあたり250.58615561894007s


 80%|████████  | 8/10 [33:26<08:23, 251.76s/it]epoch: 8, train loss: 1.24212, train accuracy: 0.4144515162860352, valid loss: 1.2381430864334106, valid accuracy: 0.4026946107784431, time: 251.92442sec
1エポックあたり250.75343826413155s


 90%|█████████ | 9/10 [37:37<04:11, 251.51s/it]epoch: 9, train loss: 1.24132, train accuracy: 0.4170722575814302, valid loss: 1.2378170490264893, valid accuracy: 0.406437125748503, time: 250.91182sec
1エポックあたり250.7710358036889s


100%|██████████| 10/10 [41:42<00:00, 250.29s/it]epoch: 10, train loss: 1.24112, train accuracy: 0.4178210408086859, valid loss: 1.237746238708496, valid accuracy: 0.405688622754491, time: 245.90981sec
1エポックあたり250.2849129676819s

train accuracy: 0.4178210408086859
test accuracy: 0.3929640718562874
"""