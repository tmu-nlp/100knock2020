"""
79. 多層ニューラルネットワークPermalink
問題78のコードを改変し，バイアス項の導入や多層化など，ニューラルネットワークの形状を変更しながら，高性能なカテゴリ分類器を構築せよ．
"""

import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
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

class MLPNet(torch.nn.Module):
    def __init__(self, input_size, mid_size, output_size, mid_layers):
        torch.manual_seed(7)
        super().__init__()
        self.mid_layers = mid_layers
        self.fc = torch.nn.Linear(input_size, mid_size, bias=False)
        self.fc_mid = torch.nn.Linear(mid_size, mid_size, bias=False)
        self.fc_out = torch.nn.Linear(mid_size, output_size, bias=False)
        #中間層の後にバッチノーマライゼーションを行う
        self.bn = torch.nn.BatchNorm1d(mid_size)

    def forward(self, x):
        x = F.relu(self.fc(x))
        for _ in range(self.mid_layers):
            x = self.fc_mid(x)
        x = F.relu(self.fc_out(x))
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
        self.model = MLPNet(d, 600, L, 10)
        self.criterion = torch.nn.CrossEntropyLoss()
        #オプティマイザ
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
    def fit(self, tr_data, va_data, test_data, epochs):
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
        print(f"1エポックあたり{total_time/epochs}s\t", end="")
        test_loss, test_acc = self.calculate_loss_acc(test_data)
        print(f"test accuracy{test_acc}\n")

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
        test_l_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        sgc[i].fit(train_l_dataset, valid_l_dataset, test_l_dataset, 10)

"""
bias: False, layer: 1, mid_size: 150
batch size: 1
epoch: 1, train loss: 0.30971, train accuracy: 0.899382253837514, valid loss: 0.34591483288540087, valid accuracy: 0.8862275449101796, time: 16.02117sec
epoch: 2, train loss: 0.28707, train accuracy: 0.9005990265818046, valid loss: 0.3607788849677906, valid accuracy: 0.8824850299401198, time: 12.48617sec
epoch: 3, train loss: 0.27602, train accuracy: 0.9085548483713964, valid loss: 0.38100327237111387, valid accuracy: 0.8809880239520959, time: 12.50383sec
epoch: 4, train loss: 0.26163, train accuracy: 0.9087420441782104, valid loss: 0.3889533145118782, valid accuracy: 0.8712574850299402, time: 12.97649sec
epoch: 5, train loss: 0.29830, train accuracy: 0.9049045301385249, valid loss: 0.5402341432748343, valid accuracy: 0.8675149700598802, time: 12.40108sec
epoch: 6, train loss: 0.20173, train accuracy: 0.932890303257207, valid loss: 0.39055891293785083, valid accuracy: 0.8869760479041916, time: 12.63972sec
epoch: 7, train loss: 0.15274, train accuracy: 0.9488019468363909, valid loss: 0.4199002196013499, valid accuracy: 0.8952095808383234, time: 13.12661sec
epoch: 8, train loss: 0.14027, train accuracy: 0.9529202545862973, valid loss: 0.49001485071923395, valid accuracy: 0.8847305389221557, time: 12.75347sec
epoch: 9, train loss: 0.12432, train accuracy: 0.9565705728191688, valid loss: 0.5266147485804997, valid accuracy: 0.8892215568862275, time: 12.67289sec
epoch: 10, train loss: 0.10561, train accuracy: 0.9650879820292025, valid loss: 0.5672599018543844, valid accuracy: 0.8832335329341318, time: 12.54911sec
1エポックあたり13.01305525302887s       test accuracy0.8974550898203593

batch size: 2
epoch: 1, train loss: 0.34927, train accuracy: 0.8820666417072257, valid loss: 0.3882060938526511, valid accuracy: 0.8764970059880239, time: 5.86392sec
epoch: 2, train loss: 0.28639, train accuracy: 0.9048109322351179, valid loss: 0.35472694716023007, valid accuracy: 0.8802395209580839, time: 5.88189sec
epoch: 3, train loss: 0.24375, train accuracy: 0.9134219393485586, valid loss: 0.3465653696383309, valid accuracy: 0.8779940119760479, time: 5.82586sec
epoch: 4, train loss: 0.19823, train accuracy: 0.9341070760014976, valid loss: 0.34090298716117284, valid accuracy: 0.8982035928143712, time: 5.91924sec
epoch: 5, train loss: 0.18360, train accuracy: 0.9415949082740547, valid loss: 0.34391921751455495, valid accuracy: 0.8877245508982036, time: 5.88395sec
epoch: 6, train loss: 0.16122, train accuracy: 0.9445900411830775, valid loss: 0.3981737947343819, valid accuracy: 0.8854790419161677, time: 5.81805sec
epoch: 7, train loss: 0.13049, train accuracy: 0.9551666042680644, valid loss: 0.4033153444441709, valid accuracy: 0.8862275449101796, time: 5.94035sec
epoch: 8, train loss: 0.13362, train accuracy: 0.95376263571696, valid loss: 0.408063201173461, valid accuracy: 0.8907185628742516, time: 5.85083sec
epoch: 9, train loss: 0.09157, train accuracy: 0.9696742792961438, valid loss: 0.5029216162983072, valid accuracy: 0.8899700598802395, time: 5.86898sec
epoch: 10, train loss: 0.08892, train accuracy: 0.9668663421939349, valid loss: 0.5421348469759426, valid accuracy: 0.8809880239520959, time: 5.94815sec
1エポックあたり5.880121803283691s       test accuracy0.8862275449101796

batch size: 4
epoch: 1, train loss: 0.30489, train accuracy: 0.8979782852864095, valid loss: 0.3298353917058836, valid accuracy: 0.8959580838323353, time: 3.73415sec
epoch: 2, train loss: 0.31507, train accuracy: 0.8875889180082366, valid loss: 0.3733192014425692, valid accuracy: 0.8772455089820359, time: 3.76955sec
epoch: 3, train loss: 0.26814, train accuracy: 0.9033133657806065, valid loss: 0.34565270531551023, valid accuracy: 0.8824850299401198, time: 4.22866sec
epoch: 4, train loss: 0.21114, train accuracy: 0.9294271808311494, valid loss: 0.31279882226065414, valid accuracy: 0.8937125748502994, time: 3.63239sec
epoch: 5, train loss: 0.19680, train accuracy: 0.932890303257207, valid loss: 0.33082035323395814, valid accuracy: 0.8854790419161677, time: 3.78946sec
epoch: 6, train loss: 0.16176, train accuracy: 0.9416885061774616, valid loss: 0.3266116153703871, valid accuracy: 0.8937125748502994, time: 3.85283sec
epoch: 7, train loss: 0.12090, train accuracy: 0.9627480344440285, valid loss: 0.35695519496290806, valid accuracy: 0.8854790419161677, time: 3.69292sec
epoch: 8, train loss: 0.10232, train accuracy: 0.9670535380007488, valid loss: 0.3858360605281959, valid accuracy: 0.8959580838323353, time: 3.73876sec
epoch: 9, train loss: 0.12001, train accuracy: 0.9590041183077499, valid loss: 0.4408421025646348, valid accuracy: 0.874251497005988, time: 3.73706sec
epoch: 10, train loss: 0.08571, train accuracy: 0.9756645451141894, valid loss: 0.4438411818723321, valid accuracy: 0.8854790419161677, time: 3.80715sec
1エポックあたり3.7982934713363647s      test accuracy0.8929640718562875

batch size: 8
epoch: 1, train loss: 0.39605, train accuracy: 0.8637214526394609, valid loss: 0.41347390271634044, valid accuracy: 0.8645209580838323, time: 2.02167sec
epoch: 2, train loss: 0.32284, train accuracy: 0.8873081242980158, valid loss: 0.35594347197685144, valid accuracy: 0.8794910179640718, time: 2.31694sec
epoch: 3, train loss: 0.26358, train accuracy: 0.912392362411082, valid loss: 0.32283143086838506, valid accuracy: 0.8974550898203593, time: 2.27532sec
epoch: 4, train loss: 0.25129, train accuracy: 0.9113627854736054, valid loss: 0.32453336796450044, valid accuracy: 0.8877245508982036, time: 2.02673sec
epoch: 5, train loss: 0.22929, train accuracy: 0.9221265443654062, valid loss: 0.32647575810551643, valid accuracy: 0.8997005988023952, time: 2.13406sec
epoch: 6, train loss: 0.20791, train accuracy: 0.9283976038936728, valid loss: 0.3212523978303917, valid accuracy: 0.8989520958083832, time: 2.33537sec
epoch: 7, train loss: 0.19026, train accuracy: 0.9346686634219393, valid loss: 0.33621256167492913, valid accuracy: 0.8937125748502994, time: 2.40690sec
epoch: 8, train loss: 0.15958, train accuracy: 0.9472107824784725, valid loss: 0.3319583237784619, valid accuracy: 0.9011976047904192, time: 2.19927sec
epoch: 9, train loss: 0.15552, train accuracy: 0.9440284537626357, valid loss: 0.34973183752059894, valid accuracy: 0.8854790419161677, time: 2.29569sec
epoch: 10, train loss: 0.11731, train accuracy: 0.9648071883189816, valid loss: 0.3523497188548853, valid accuracy: 0.8922155688622755, time: 2.23328sec
1エポックあたり2.224520945549011s       test accuracy0.8959580838323353
"""

"""
bias: True, layer: 1, mid_size: 150
batch size: 1
epoch: 1, train loss: 0.51603, train accuracy: 0.8026020217147136, valid loss: 0.5351231305987655, valid accuracy: 0.7926646706586826, time: 15.51402sec
epoch: 2, train loss: 0.49969, train accuracy: 0.8077499064020965, valid loss: 0.5431873132171728, valid accuracy: 0.7971556886227545, time: 15.87213sec
epoch: 3, train loss: 0.47191, train accuracy: 0.8205728191688506, valid loss: 0.5198814978584401, valid accuracy: 0.8158682634730539, time: 15.21648sec
epoch: 4, train loss: 0.46004, train accuracy: 0.8199176338450018, valid loss: 0.5255471296108482, valid accuracy: 0.8023952095808383, time: 14.49113sec
epoch: 5, train loss: 0.37157, train accuracy: 0.8877761138150505, valid loss: 0.4371942542361299, valid accuracy: 0.8794910179640718, time: 14.43481sec
epoch: 6, train loss: 0.36885, train accuracy: 0.8832834144515163, valid loss: 0.4446434124957937, valid accuracy: 0.8697604790419161, time: 14.98193sec
epoch: 7, train loss: 0.25628, train accuracy: 0.9140771246724073, valid loss: 0.37948927129934057, valid accuracy: 0.8862275449101796, time: 14.77667sec
epoch: 8, train loss: 0.31091, train accuracy: 0.8913328341445151, valid loss: 0.4100396581802975, valid accuracy: 0.8667664670658682, time: 16.10594sec
epoch: 9, train loss: 0.33239, train accuracy: 0.8974166978659678, valid loss: 0.49869531590336086, valid accuracy: 0.8705089820359282, time: 14.97411sec
epoch: 10, train loss: 0.30299, train accuracy: 0.9108011980531636, valid loss: 0.475967912571055, valid accuracy: 0.8779940119760479, time: 14.46179sec
1エポックあたり15.082902002334595s      test accuracy0.8690119760479041

batch size: 2
epoch: 1, train loss: 0.50501, train accuracy: 0.8141145638337701, valid loss: 0.5211892553758849, valid accuracy: 0.8106287425149701, time: 7.93240sec
epoch: 2, train loss: 0.34873, train accuracy: 0.8819730438038188, valid loss: 0.3702112806729338, valid accuracy: 0.8720059880239521, time: 6.50438sec
epoch: 3, train loss: 0.31769, train accuracy: 0.8937663796330962, valid loss: 0.36810182021637333, valid accuracy: 0.8832335329341318, time: 6.50098sec
epoch: 4, train loss: 0.35322, train accuracy: 0.8702733058779484, valid loss: 0.40975152825604344, valid accuracy: 0.8495508982035929, time: 6.58068sec
epoch: 5, train loss: 0.28783, train accuracy: 0.9015350056158742, valid loss: 0.3631289289661978, valid accuracy: 0.8832335329341318, time: 6.63279sec
epoch: 6, train loss: 0.25969, train accuracy: 0.9052789217521527, valid loss: 0.356373603635933, valid accuracy: 0.8787425149700598, time: 6.56452sec
epoch: 7, train loss: 0.27103, train accuracy: 0.9010670160988394, valid loss: 0.38220089969802995, valid accuracy: 0.8675149700598802, time: 6.63619sec
epoch: 8, train loss: 0.25753, train accuracy: 0.9055597154623737, valid loss: 0.37564487228954696, valid accuracy: 0.8712574850299402, time: 6.61002sec
epoch: 9, train loss: 0.25766, train accuracy: 0.9119243728940472, valid loss: 0.429581300087011, valid accuracy: 0.8787425149700598, time: 6.65418sec
epoch: 10, train loss: 0.17283, train accuracy: 0.9422500935979035, valid loss: 0.3462034972667915, valid accuracy: 0.8914670658682635, time: 6.47545sec
1エポックあたり6.709158563613892s       test accuracy0.8937125748502994

batch size: 4
epoch: 1, train loss: 0.54273, train accuracy: 0.7974541370273306, valid loss: 0.5489726941746765, valid accuracy: 0.7941616766467066, time: 4.35955sec
epoch: 2, train loss: 0.39946, train accuracy: 0.8527704979408461, valid loss: 0.41418805247741547, valid accuracy: 0.8517964071856288, time: 6.95713sec
epoch: 3, train loss: 0.28679, train accuracy: 0.9015350056158742, valid loss: 0.33538170881221885, valid accuracy: 0.8974550898203593, time: 4.87415sec
epoch: 4, train loss: 0.29318, train accuracy: 0.8970423062523399, valid loss: 0.362317686367391, valid accuracy: 0.8974550898203593, time: 7.70201sec
epoch: 5, train loss: 0.26036, train accuracy: 0.9100524148259079, valid loss: 0.3483776501323268, valid accuracy: 0.8952095808383234, time: 4.90022sec
epoch: 6, train loss: 0.24997, train accuracy: 0.9105204043429427, valid loss: 0.3484309458670145, valid accuracy: 0.8877245508982036, time: 5.87414sec
epoch: 7, train loss: 0.34050, train accuracy: 0.8745788094346687, valid loss: 0.4498229888560596, valid accuracy: 0.8615269461077845, time: 4.27529sec
epoch: 8, train loss: 0.21216, train accuracy: 0.9256832646948708, valid loss: 0.3442841718347344, valid accuracy: 0.8907185628742516, time: 4.22541sec
epoch: 9, train loss: 0.18318, train accuracy: 0.9340134780980907, valid loss: 0.32675867724723134, valid accuracy: 0.8929640718562875, time: 4.31339sec
epoch: 10, train loss: 0.19998, train accuracy: 0.9290527892175215, valid loss: 0.3474975977527622, valid accuracy: 0.8907185628742516, time: 4.21136sec
1エポックあたり5.169265818595886s       test accuracy0.8839820359281437

batch size: 8
epoch: 1, train loss: 0.58037, train accuracy: 0.8124298015724448, valid loss: 0.5883161724506024, valid accuracy: 0.8068862275449101, time: 2.36183sec
epoch: 2, train loss: 0.46827, train accuracy: 0.8228191688506178, valid loss: 0.49323532331601055, valid accuracy: 0.8196107784431138, time: 2.85749sec
epoch: 3, train loss: 0.32290, train accuracy: 0.8897416697865967, valid loss: 0.3543047458825711, valid accuracy: 0.8884730538922155, time: 4.07375sec
epoch: 4, train loss: 0.29775, train accuracy: 0.9005054286783976, valid loss: 0.35174841905280413, valid accuracy: 0.8892215568862275, time: 4.12845sec
epoch: 5, train loss: 0.25479, train accuracy: 0.9118307749906402, valid loss: 0.31990562813791507, valid accuracy: 0.8914670658682635, time: 3.50204sec
epoch: 6, train loss: 0.24846, train accuracy: 0.9125795582178959, valid loss: 0.32549365017072346, valid accuracy: 0.8907185628742516, time: 2.35104sec
epoch: 7, train loss: 0.25976, train accuracy: 0.9058405091725945, valid loss: 0.36163491746522264, valid accuracy: 0.8824850299401198, time: 2.38259sec
epoch: 8, train loss: 0.21500, train accuracy: 0.9256832646948708, valid loss: 0.32248780172050534, valid accuracy: 0.8914670658682635, time: 2.31124sec
epoch: 9, train loss: 0.24371, train accuracy: 0.9089292399850243, valid loss: 0.37190501945383264, valid accuracy: 0.8764970059880239, time: 2.45194sec
epoch: 10, train loss: 0.19469, train accuracy: 0.9323287158367652, valid loss: 0.33096738870138537, valid accuracy: 0.8899700598802395, time: 2.27003sec
1エポックあたり2.86903977394104s        test accuracy0.8899700598802395
"""

"""
bias: False, layer: 1, mid_size: 400
batch size: 1
epoch: 1, train loss: 0.32313, train accuracy: 0.8911456383377012, valid loss: 0.3921719504301832, valid accuracy: 0.875, time: 20.05094sec
epoch: 2, train loss: 0.30500, train accuracy: 0.8824410333208537, valid loss: 0.3871239557129475, valid accuracy: 0.8577844311377245, time: 20.01260sec
epoch: 3, train loss: 0.29217, train accuracy: 0.8941407712467241, valid loss: 0.42568097183421966, valid accuracy: 0.8585329341317365, time: 19.93053sec
epoch: 4, train loss: 0.23052, train accuracy: 0.922500935979034, valid loss: 0.4161707033743937, valid accuracy: 0.8794910179640718, time: 20.23755sec
epoch: 5, train loss: 0.17193, train accuracy: 0.9506739049045302, valid loss: 0.35575569101623095, valid accuracy: 0.8899700598802395, time: 20.04732sec
epoch: 6, train loss: 0.16134, train accuracy: 0.9465555971546238, valid loss: 0.46361909228585574, valid accuracy: 0.8944610778443114, time: 20.28881sec
epoch: 7, train loss: 0.14410, train accuracy: 0.9546986147510296, valid loss: 0.4680646458709513, valid accuracy: 0.8937125748502994, time: 19.99153sec
epoch: 8, train loss: 0.14622, train accuracy: 0.9469299887682516, valid loss: 0.503712308157065, valid accuracy: 0.875, time: 19.98196sec
epoch: 9, train loss: 0.16397, train accuracy: 0.9435604642456009, valid loss: 0.534906045061967, valid accuracy: 0.8660179640718563, time: 20.16868sec
epoch: 10, train loss: 0.06547, train accuracy: 0.9758517409210034, valid loss: 0.5104952454570858, valid accuracy: 0.8974550898203593, time: 20.09525sec
1エポックあたり20.080516266822816s      test accuracy0.9026946107784432

batch size: 2
epoch: 1, train loss: 0.37097, train accuracy: 0.8755147884687383, valid loss: 0.40080705836217706, valid accuracy: 0.8615269461077845, time: 11.10059sec
epoch: 2, train loss: 0.26401, train accuracy: 0.9134219393485586, valid loss: 0.3308919419913475, valid accuracy: 0.8952095808383234, time: 11.21436sec
epoch: 3, train loss: 0.20370, train accuracy: 0.9311119430924747, valid loss: 0.3201986425825122, valid accuracy: 0.8929640718562875, time: 11.09942sec
epoch: 4, train loss: 0.16811, train accuracy: 0.9423436915013104, valid loss: 0.326269466127348, valid accuracy: 0.8937125748502994, time: 11.05812sec
epoch: 5, train loss: 0.13708, train accuracy: 0.9564769749157619, valid loss: 0.320608475261558, valid accuracy: 0.9086826347305389, time: 11.37246sec
epoch: 6, train loss: 0.17698, train accuracy: 0.9395357543991014, valid loss: 0.44676412005078525, valid accuracy: 0.8779940119760479, time: 11.05019sec
epoch: 7, train loss: 0.10747, train accuracy: 0.9627480344440285, valid loss: 0.4441797439559833, valid accuracy: 0.8824850299401198, time: 11.15568sec
epoch: 8, train loss: 0.14641, train accuracy: 0.9554473979782853, valid loss: 0.463953434976585, valid accuracy: 0.875, time: 11.02216sec
epoch: 9, train loss: 0.07068, train accuracy: 0.976600524148259, valid loss: 0.45510899154141937, valid accuracy: 0.8952095808383234, time: 11.38392sec
epoch: 10, train loss: 0.04162, train accuracy: 0.987270685136653, valid loss: 0.45601004331239575, valid accuracy: 0.8914670658682635, time: 11.24293sec
1エポックあたり11.169982433319092s      test accuracy0.9086826347305389

batch size: 4
epoch: 1, train loss: 0.30905, train accuracy: 0.8923624110819918, valid loss: 0.3382328195740681, valid accuracy: 0.8899700598802395, time: 5.94264sec
epoch: 2, train loss: 0.27043, train accuracy: 0.9080868588543617, valid loss: 0.33428672597740783, valid accuracy: 0.8914670658682635, time: 5.93786sec
epoch: 3, train loss: 0.23033, train accuracy: 0.9208161737177087, valid loss: 0.3158077449486577, valid accuracy: 0.8884730538922155, time: 5.87748sec
epoch: 4, train loss: 0.19163, train accuracy: 0.9357918382628229, valid loss: 0.31327994612163135, valid accuracy: 0.8952095808383234, time: 5.85892sec
epoch: 5, train loss: 0.15876, train accuracy: 0.95376263571696, valid loss: 0.3247585403474594, valid accuracy: 0.8952095808383234, time: 6.24194sec
epoch: 6, train loss: 0.12874, train accuracy: 0.9619056533133657, valid loss: 0.3316710511353449, valid accuracy: 0.8989520958083832, time: 6.06434sec
epoch: 7, train loss: 0.10468, train accuracy: 0.9687383002620741, valid loss: 0.3628798318341232, valid accuracy: 0.8839820359281437, time: 5.91543sec
epoch: 8, train loss: 0.09422, train accuracy: 0.969299887682516, valid loss: 0.39858464943784216, valid accuracy: 0.8884730538922155, time: 5.86521sec
epoch: 9, train loss: 0.07347, train accuracy: 0.9751965555971546, valid loss: 0.44724534500058893, valid accuracy: 0.8869760479041916, time: 5.86403sec
epoch: 10, train loss: 0.06161, train accuracy: 0.9817484088356421, valid loss: 0.5035670037344341, valid accuracy: 0.8914670658682635, time: 5.95317sec
1エポックあたり5.952103090286255s       test accuracy0.8929640718562875

batch size: 8
epoch: 1, train loss: 0.36549, train accuracy: 0.8795394983152377, valid loss: 0.38514277682511394, valid accuracy: 0.8764970059880239, time: 3.42571sec
epoch: 2, train loss: 0.29667, train accuracy: 0.8955447397978286, valid loss: 0.33274595672543533, valid accuracy: 0.8899700598802395, time: 3.48784sec
epoch: 3, train loss: 0.29059, train accuracy: 0.8948895544739798, valid loss: 0.34486764641534423, valid accuracy: 0.8705089820359282, time: 3.30875sec
epoch: 4, train loss: 0.23660, train accuracy: 0.9194122051666043, valid loss: 0.31712709718821624, valid accuracy: 0.8982035928143712, time: 3.52335sec
epoch: 5, train loss: 0.21292, train accuracy: 0.9299887682515912, valid loss: 0.32035669687756163, valid accuracy: 0.8982035928143712, time: 3.44970sec
epoch: 6, train loss: 0.18414, train accuracy: 0.9423436915013104, valid loss: 0.31526787415102214, valid accuracy: 0.8974550898203593, time: 3.33623sec
epoch: 7, train loss: 0.14945, train accuracy: 0.9525458629726694, valid loss: 0.32201065296658543, valid accuracy: 0.9011976047904192, time: 3.39086sec
epoch: 8, train loss: 0.13069, train accuracy: 0.963871209284912, valid loss: 0.314032508216486, valid accuracy: 0.8982035928143712, time: 3.50271sec
epoch: 9, train loss: 0.12568, train accuracy: 0.9573193560464246, valid loss: 0.37204203608447217, valid accuracy: 0.8967065868263473, time: 3.27868sec
epoch: 10, train loss: 0.08908, train accuracy: 0.9721078247847248, valid loss: 0.37670853763765066, valid accuracy: 0.9004491017964071, time: 3.41799sec
1エポックあたり3.412181758880615s       test accuracy0.8967065868263473
"""

"""
bias: False, layer: 2, mid_size: 400
batch size: 1
epoch: 1, train loss: 0.38269, train accuracy: 0.8728004492699364, valid loss: 0.4453469924147407, valid accuracy: 0.8517964071856288, time: 25.03592sec
epoch: 2, train loss: 0.44968, train accuracy: 0.8367652564582553, valid loss: 0.5122261511829739, valid accuracy: 0.813622754491018, time: 25.26105sec
epoch: 3, train loss: 0.30252, train accuracy: 0.9043429427180831, valid loss: 0.401306338547403, valid accuracy: 0.875, time: 25.44560sec
epoch: 4, train loss: 0.48547, train accuracy: 0.8485585922875327, valid loss: 0.5975412153734815, valid accuracy: 0.8383233532934131, time: 25.19023sec
epoch: 5, train loss: 0.39042, train accuracy: 0.8630662673156121, valid loss: 0.5582860758752348, valid accuracy: 0.8278443113772455, time: 24.80105sec
epoch: 6, train loss: 1.38629, train accuracy: 0.06813927368026956, valid loss: 1.3862943649291992, valid accuracy: 0.06811377245508982, time: 24.81597sec
epoch: 7, train loss: 1.38629, train accuracy: 0.06813927368026956, valid loss: 1.3862943649291992, valid accuracy: 0.06811377245508982, time: 25.10941sec
epoch: 8, train loss: 1.38629, train accuracy: 0.06813927368026956, valid loss: 1.3862943649291992, valid accuracy: 0.06811377245508982, time: 24.77350sec
epoch: 9, train loss: 1.38629, train accuracy: 0.06813927368026956, valid loss: 1.3862943649291992, valid accuracy: 0.06811377245508982, time: 24.85768sec
epoch: 10, train loss: 1.38629, train accuracy: 0.06813927368026956, valid loss: 1.3862943649291992, valid accuracy: 0.06811377245508982, time: 26.60199sec
1エポックあたり25.18923988342285s       test accuracy0.06811377245508982

batch size: 2
epoch: 1, train loss: 0.44662, train accuracy: 0.8668101834518906, valid loss: 0.47743397158895173, valid accuracy: 0.8555389221556886, time: 18.90140sec
epoch: 2, train loss: 0.28789, train accuracy: 0.9056533133657806, valid loss: 0.348300884016969, valid accuracy: 0.8862275449101796, time: 14.48267sec
epoch: 3, train loss: 0.22055, train accuracy: 0.927836016473231, valid loss: 0.3476116049255229, valid accuracy: 0.8892215568862275, time: 14.63875sec
epoch: 4, train loss: 0.19348, train accuracy: 0.9352302508423811, valid loss: 0.3653142904230525, valid accuracy: 0.8944610778443114, time: 14.31780sec
epoch: 5, train loss: 0.21185, train accuracy: 0.9267128416323475, valid loss: 0.40928366977491276, valid accuracy: 0.8772455089820359, time: 14.32183sec
epoch: 6, train loss: 0.18345, train accuracy: 0.930363159865219, valid loss: 0.45056117951986063, valid accuracy: 0.875748502994012, time: 14.27343sec
epoch: 7, train loss: 0.10696, train accuracy: 0.9644327967053538, valid loss: 0.42859974514683385, valid accuracy: 0.8884730538922155, time: 14.28327sec
epoch: 8, train loss: 0.14654, train accuracy: 0.953481842006739, valid loss: 0.4695110525117854, valid accuracy: 0.8675149700598802, time: 18.34832sec
epoch: 9, train loss: 0.07630, train accuracy: 0.976787719955073, valid loss: 0.5030319697690792, valid accuracy: 0.8869760479041916, time: 17.92497sec
epoch: 10, train loss: 0.07460, train accuracy: 0.974073380756271, valid loss: 0.508096731976789, valid accuracy: 0.8914670658682635, time: 18.17776sec
1エポックあたり15.967019414901733s      test accuracy0.8967065868263473

batch size: 4
epoch: 1, train loss: 0.32308, train accuracy: 0.8888992886559341, valid loss: 0.3519908081688556, valid accuracy: 0.8884730538922155, time: 8.40779sec
epoch: 2, train loss: 0.28126, train accuracy: 0.9052789217521527, valid loss: 0.3477563243927462, valid accuracy: 0.8869760479041916, time: 8.43420sec
epoch: 3, train loss: 0.22893, train accuracy: 0.922500935979034, valid loss: 0.33213463971842866, valid accuracy: 0.8907185628742516, time: 8.07999sec
epoch: 4, train loss: 0.18341, train accuracy: 0.9415013103706477, valid loss: 0.3272132538369857, valid accuracy: 0.9034431137724551, time: 7.39562sec
epoch: 5, train loss: 0.15297, train accuracy: 0.9536690378135529, valid loss: 0.3442504327886671, valid accuracy: 0.8989520958083832, time: 7.52591sec
epoch: 6, train loss: 0.14019, train accuracy: 0.9533882441033321, valid loss: 0.37827758941180334, valid accuracy: 0.8929640718562875, time: 7.30485sec
epoch: 7, train loss: 0.12790, train accuracy: 0.9587233245975291, valid loss: 0.40360745383576163, valid accuracy: 0.875, time: 8.68002sec
epoch: 8, train loss: 0.08947, train accuracy: 0.9706102583302134, valid loss: 0.4217814158100587, valid accuracy: 0.8929640718562875, time: 8.53344sec
epoch: 9, train loss: 0.06943, train accuracy: 0.9762261325346312, valid loss: 0.5167592054528541, valid accuracy: 0.8907185628742516, time: 8.83868sec
epoch: 10, train loss: 0.06608, train accuracy: 0.979314863347061, valid loss: 0.5400408216720242, valid accuracy: 0.8944610778443114, time: 7.88111sec
1エポックあたり8.10815978050232s        test accuracy0.8989520958083832

batch size: 8
epoch: 1, train loss: 0.39664, train accuracy: 0.865967802321228, valid loss: 0.4213201277136446, valid accuracy: 0.8637724550898204, time: 5.17888sec
epoch: 2, train loss: 0.28803, train accuracy: 0.9018157993260951, valid loss: 0.3311131064205648, valid accuracy: 0.8914670658682635, time: 4.43228sec
epoch: 3, train loss: 0.31482, train accuracy: 0.8889928865593411, valid loss: 0.37885909127127265, valid accuracy: 0.8645209580838323, time: 4.27428sec
epoch: 4, train loss: 0.19760, train accuracy: 0.9344814676151254, valid loss: 0.31040807704937584, valid accuracy: 0.9011976047904192, time: 4.21106sec
epoch: 5, train loss: 0.17065, train accuracy: 0.9490827405466118, valid loss: 0.333242099721007, valid accuracy: 0.8922155688622755, time: 4.26985sec
epoch: 6, train loss: 0.13461, train accuracy: 0.961344065892924, valid loss: 0.32683359462667455, valid accuracy: 0.8982035928143712, time: 4.13842sec
epoch: 7, train loss: 0.10518, train accuracy: 0.9674279296143766, valid loss: 0.3563985535062844, valid accuracy: 0.8997005988023952, time: 4.26760sec
epoch: 8, train loss: 0.09785, train accuracy: 0.9734181954324224, valid loss: 0.3682024698247185, valid accuracy: 0.8884730538922155, time: 4.22619sec
epoch: 9, train loss: 0.08813, train accuracy: 0.9720142268813179, valid loss: 0.42777140286380233, valid accuracy: 0.8989520958083832, time: 4.46083sec
epoch: 10, train loss: 0.04720, train accuracy: 0.9881130662673157, valid loss: 0.4355563497296981, valid accuracy: 0.8989520958083832, time: 4.30363sec
1エポックあたり4.37630250453949s        test accuracy0.9011976047904192
"""