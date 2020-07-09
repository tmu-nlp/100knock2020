"""
77. ミニバッチ化Permalink
問題76のコードを改変し，B事例ごとに損失・勾配を計算し，行列Wの値を更新せよ（ミニバッチ化）．
Bの値を1,2,4,8,…と変化させながら，1エポックの学習に要する時間を比較せよ．
"""

import time
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from matplotlib import pyplot as plt

X_train_data = "X_train.npy"
Y_train_data = "Y_train.npy"
X_valid_data = "X_valid.npy"
Y_valid_data = "Y_valid.npy"
X_test_data = "X_test.npy"
Y_test_data = "Y_test.npy"

X_train = np.load(file=X_train_data)
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = np.load(file=Y_train_data)
Y_train = torch.tensor(Y_train, dtype=torch.int64)
X_valid = np.load(file=X_valid_data)
X_valid = torch.tensor(X_valid, dtype=torch.float32)
Y_valid = np.load(file=Y_valid_data)
Y_valid = torch.tensor(Y_valid, dtype=torch.int64)
X_test = np.load(file=X_test_data)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = np.load(file=Y_test_data)
Y_test = torch.tensor(Y_test, dtype=torch.int64)

d = 300
L = 4

class SLPNet(torch.nn.Module):
    def __init__(self, input_size, output_size):
        torch.manual_seed(7)
        #nn.Moduleのinitを引っ張ってくる
        super().__init__()
        self.fc = torch.nn.Linear(input_size, output_size, bias=False)
        #重みを平均0分散1で初期化
        torch.nn.init.normal_(self.fc.weight)

    def forward(self, x):
        x = self.fc(x)
        return x

#データセットをlistみたいな形にしておけるクラス
class CreateData():
    def __init__(self, x_data, y_data):
        self.x = x_data
        self.y = y_data

    def __len__(self):                      #len()でサイズを返す
        return len(self.y)

    def __getitem__(self, idx):             #getitem()で指定されたインデックスの要素を返す
        return [self.x[idx], self.y[idx]]

class SGC():
    def __init__(self):
        self.model = SLPNet(d, L)
        self.criterion = torch.nn.CrossEntropyLoss()
        #オプティマイザ
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
    def fit(self, tr_data, va_data, epochs):
        train_log = []
        valid_log = []
        total_time = 0
        for i in range(epochs):

            s_time = time.time()

            self.model.train()
            total_loss=0.0
            for j, (sorce, target) in enumerate(tr_data):
                self.optimizer.zero_grad()          #勾配の初期化

                output = self.model.forward(sorce)  #順伝播
                loss = self.criterion(output, target)
                loss.backward()                     #逆伝播
                self.optimizer.step()               #重み更新

                total_loss += loss.item()
            train_loss = total_loss/j              #バッチ単位のロス

            train_loss, train_acc = self.calculate_loss_acc(tr_data)
            valid_loss, valid_acc = self.calculate_loss_acc(va_data)
            train_log.append([train_loss, train_acc])
            valid_log.append([valid_loss, valid_acc])

            torch.save({"epoch": i, "model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict()}, f'checkpoint{i+1}.pt')

            e_time = time.time()

            print(f"epoch: {i+1}, train loss: {train_loss:.5f}, train accuracy: {train_acc}, valid loss: {valid_loss}, valid accuracy: {valid_acc}, time: {(e_time-s_time):.5f}sec")

            total_time += e_time-s_time
        print(f"1エポックあたり{total_time/epochs}s\n")

    def calculate_loss_acc(self, data):
        self.model.eval()
        loss = 0
        total = 0
        cor = 0
        with torch.no_grad():
            for sorce, target in data:
                output = self.model(sorce)
                loss += self.criterion(output, target).item()
                pred = torch.argmax(output, dim=-1)
                total += len(sorce)
                cor += (pred == target).sum().item()
        return loss/len(data), cor/total

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
    train_dataset = CreateData(X_train, Y_train)
    valid_dataset = CreateData(X_valid, Y_valid)
    test_dataset = CreateData(X_test, Y_test)

    lis = [1, 2, 4, 8]
    sgc = [SGC(), SGC(), SGC(), SGC()]
    for i, batch_size in enumerate(lis):
        print(f"batch size: {batch_size}")
        train_l_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_l_dataset = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
        sgc[i].fit(train_l_dataset, valid_l_dataset, 10)

"""
batch size: 1
epoch: 1, train loss: 0.39711, train accuracy: 0.865967802321228, valid loss: 0.40909145980577694, valid accuracy: 0.8682634730538922, time: 8.83177sec
epoch: 2, train loss: 0.34215, train accuracy: 0.884032197678772, valid loss: 0.357929669541402, valid accuracy: 0.8884730538922155, time: 7.57207sec
epoch: 3, train loss: 0.32020, train accuracy: 0.8906776488206665, valid loss: 0.3419684574399358, valid accuracy: 0.8907185628742516, time: 8.30864sec
epoch: 4, train loss: 0.30632, train accuracy: 0.8975102957693748, valid loss: 0.3345496162609041, valid accuracy: 0.8967065868263473, time: 7.08786sec
epoch: 5, train loss: 0.29902, train accuracy: 0.8974166978659678, valid loss: 0.3298024634366435, valid accuracy: 0.8884730538922155, time: 7.68039sec
epoch: 6, train loss: 0.28753, train accuracy: 0.9027517783601647, valid loss: 0.3251913392160966, valid accuracy: 0.8944610778443114, time: 8.27797sec
epoch: 7, train loss: 0.28240, train accuracy: 0.904623736428304, valid loss: 0.3226302085471679, valid accuracy: 0.9011976047904192, time: 7.54163sec
epoch: 8, train loss: 0.28076, train accuracy: 0.9043429427180831, valid loss: 0.324737798257148, valid accuracy: 0.8959580838323353, time: 7.86980sec
epoch: 9, train loss: 0.27722, train accuracy: 0.9052789217521527, valid loss: 0.3241058335342271, valid accuracy: 0.9004491017964071, time: 6.95513sec
epoch: 10, train loss: 0.27172, train accuracy: 0.9064956944964433, valid loss: 0.3205297349356073, valid accuracy: 0.8937125748502994, time: 7.98352sec
1エポックあたり7.810877537727356s

batch size: 2
epoch: 1, train loss: 0.45526, train accuracy: 0.8450954698614751, valid loss: 0.46644790703622474, valid accuracy: 0.8390718562874252, time: 3.95947sec
epoch: 2, train loss: 0.39100, train accuracy: 0.8698053163609135, valid loss: 0.4017623611602132, valid accuracy: 0.8660179640718563, time: 3.51295sec
epoch: 3, train loss: 0.36071, train accuracy: 0.8782291276675402, valid loss: 0.37392974589933214, valid accuracy: 0.8779940119760479, time: 3.17933sec
epoch: 4, train loss: 0.34207, train accuracy: 0.8842193934855859, valid loss: 0.3598846927447967, valid accuracy: 0.8832335329341318, time: 3.22202sec
epoch: 5, train loss: 0.32882, train accuracy: 0.8888992886559341, valid loss: 0.34838802662434426, valid accuracy: 0.8914670658682635, time: 4.34814sec
epoch: 6, train loss: 0.31960, train accuracy: 0.8903032572070385, valid loss: 0.3422043491278644, valid accuracy: 0.8899700598802395, time: 2.90435sec
epoch: 7, train loss: 0.31105, train accuracy: 0.8936727817296892, valid loss: 0.3364778076186607, valid accuracy: 0.8899700598802395, time: 3.05654sec
epoch: 8, train loss: 0.30729, train accuracy: 0.8944215649569449, valid loss: 0.3348143655834184, valid accuracy: 0.8907185628742516, time: 3.09714sec
epoch: 9, train loss: 0.29972, train accuracy: 0.8973230999625609, valid loss: 0.3295917465663212, valid accuracy: 0.8899700598802395, time: 2.95461sec
epoch: 10, train loss: 0.29473, train accuracy: 0.9004118307749907, valid loss: 0.3272820479725557, valid accuracy: 0.8952095808383234, time: 3.09279sec
1エポックあたり3.332733702659607s

batch size: 4
epoch: 1, train loss: 0.54815, train accuracy: 0.8111194309247473, valid loss: 0.5626218029894604, valid accuracy: 0.7904191616766467, time: 1.59670sec
epoch: 2, train loss: 0.45627, train accuracy: 0.8447210782478473, valid loss: 0.46607050695595986, valid accuracy: 0.8420658682634731, time: 1.61236sec
epoch: 3, train loss: 0.41534, train accuracy: 0.8597903406963684, valid loss: 0.42439594815490755, valid accuracy: 0.8600299401197605, time: 1.70282sec
epoch: 4, train loss: 0.39071, train accuracy: 0.8699925121677274, valid loss: 0.40126172831967966, valid accuracy: 0.8660179640718563, time: 1.58367sec
epoch: 5, train loss: 0.37321, train accuracy: 0.8754211905653313, valid loss: 0.38564401312625246, valid accuracy: 0.8697604790419161, time: 1.59423sec
epoch: 6, train loss: 0.36022, train accuracy: 0.8796330962186447, valid loss: 0.37376199490153705, valid accuracy: 0.875, time: 1.72123sec
epoch: 7, train loss: 0.35014, train accuracy: 0.8826282291276676, valid loss: 0.36506642915118864, valid accuracy: 0.8824850299401198, time: 1.57280sec
epoch: 8, train loss: 0.34173, train accuracy: 0.8845937850992138, valid loss: 0.35800257004577174, valid accuracy: 0.8839820359281437, time: 1.58487sec
epoch: 9, train loss: 0.33452, train accuracy: 0.8859977536503182, valid loss: 0.3531765580919019, valid accuracy: 0.8862275449101796, time: 1.71659sec
epoch: 10, train loss: 0.32834, train accuracy: 0.8885248970423063, valid loss: 0.3479511697740367, valid accuracy: 0.8899700598802395, time: 1.60079sec
1エポックあたり1.6286059856414794s

batch size: 8
epoch: 1, train loss: 0.66379, train accuracy: 0.7662860351928117, valid loss: 0.6802032915596476, valid accuracy: 0.7514970059880239, time: 0.92253sec
epoch: 2, train loss: 0.54629, train accuracy: 0.8140209659303631, valid loss: 0.5600788925668436, valid accuracy: 0.7926646706586826, time: 0.91258sec
epoch: 3, train loss: 0.48990, train accuracy: 0.8333021340321977, valid loss: 0.5011731863111079, valid accuracy: 0.8218562874251497, time: 0.92641sec
epoch: 4, train loss: 0.45556, train accuracy: 0.8466866342193935, valid loss: 0.465778134450941, valid accuracy: 0.8405688622754491, time: 1.02816sec
epoch: 5, train loss: 0.43217, train accuracy: 0.8540808685885436, valid loss: 0.44163162899231484, valid accuracy: 0.8540419161676647, time: 0.91851sec
epoch: 6, train loss: 0.41489, train accuracy: 0.8613815050542868, valid loss: 0.42456725953581803, valid accuracy: 0.8592814371257484, time: 0.92772sec
epoch: 7, train loss: 0.40145, train accuracy: 0.865967802321228, valid loss: 0.41106424869936026, valid accuracy: 0.8645209580838323, time: 0.91472sec
epoch: 8, train loss: 0.39042, train accuracy: 0.8697117184575065, valid loss: 0.4005558101270727, valid accuracy: 0.8682634730538922, time: 0.92689sec
epoch: 9, train loss: 0.38114, train accuracy: 0.8728940471733433, valid loss: 0.3926192418097736, valid accuracy: 0.8675149700598802, time: 0.98710sec
epoch: 10, train loss: 0.37337, train accuracy: 0.8741108199176338, valid loss: 0.38549793345441, valid accuracy: 0.8735029940119761, time: 0.94054sec
1エポックあたり0.9405147790908813s
"""