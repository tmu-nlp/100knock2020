"""
81. RNNによる予測Permalink
ID番号で表現された単語列x=(x1,x2,…,xT)がある．
ただし，Tは単語列の長さ，xt∈ℝVは単語のID番号のone-hot表記である（Vは単語の総数である）．
再帰型ニューラルネットワーク（RNN: Recurrent Neural Network）を用い，単語列xからカテゴリyを予測するモデルとして，次式を実装せよ．

h→0=0,h→t=RNN−→−−(emb(xt),h→t−1),y=softmax(W(yh)h→T+b(y))

ただし，emb(x)∈ℝdwは単語埋め込み（単語のone-hot表記から単語ベクトルに変換する関数），
h→t∈ℝdhは時刻tの隠れ状態ベクトル，RNN−→−−(x,h)は入力xと前時刻の隠れ状態hから次状態を計算するRNNユニット，
W(yh)∈ℝL×dhは隠れ状態ベクトルからカテゴリを予測するための行列，
b(y)∈ℝLはバイアス項である（dw,dh,Lはそれぞれ，単語埋め込みの次元数，隠れ状態ベクトルの次元数，ラベル数である）．
RNNユニットRNN−→−−(x,h)には様々な構成が考えられるが，典型例として次式が挙げられる．

RNN−→−−(x,h)=g(W(hx)x+W(hh)h+b(h))

ただし，W(hx)∈ℝdh×dw，W(hh)∈ℝdh×dh,b(h)∈ℝdhはRNNユニットのパラメータ，gは活性化関数（例えばtanhやReLUなど）である．

なお，この問題ではパラメータの学習を行わず，ランダムに初期化されたパラメータでyを計算するだけでよい．
次元数などのハイパーパラメータは，dw=300,dh=50など，適当な値に設定せよ（以降の問題でも同様である）．
"""

import csv
from knock80 import translate_to_id
import numpy as np
import torch
from torch import nn


train_path = "../chapter06/train.csv"
valid_path = "../chapter06/valid.csv"
test_path = "../chapter06/test.csv"

class CreateData():
    def __init__(self, x_data, y_data, to_idvec):
        self.X = x_data
        self.Y = y_data
        self.to_idvec = to_idvec

    def __len__(self):                      #len()でサイズを返す
        return len(self.Y)

    def __getitem__(self, idx):             #getitem()で指定されたインデックスの要素を返す
        id_text, V = self.to_idvec(self.X[idx])
        id_text = id_text.split()
        id_text = [int(id) for id in id_text]
        return {
            "id_text": torch.tensor(id_text, dtype=torch.int64),
            "labels": torch.tensor(self.Y[idx], dtype=torch.int64)
        }

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
        out, hidden = self.rnn(emb, hidden)

        out = self.fc(out[:, -1])
        return out

    def init_hidden(self):
        #batch_size*hidden_sizeを1つ
        hidden = torch.zeros(1, self.batch_size, self.hidden_size)
        return hidden

def get_X_Y(file_path):
    with open(file_path) as f:
        reader = csv.reader(f, delimiter="\t")
        l = [row for row in reader]
        l = l[1:]
        category = ["b", "t", "e", "m"]
        X = []
        Y = []
        for i, row in enumerate(l):
            X.append(row[0])
            Y.append(category.index(row[1]))
    return X, Y

"""
def make_vec(id_text):
    vec = []
    for id in id_text.split():
        x = [0]*V
        x[int(id)] = 1
        vec.append(x)
    return torch.tensor(vec)
"""

if __name__ =="__main__":
    _, V = translate_to_id("")
    VOCAB_SIZE = V+1
    EMB_SIZE = 300
    PADDING_IDX = V
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50

    train_X, train_Y = get_X_Y(train_path)
    valid_X, valid_Y = get_X_Y(valid_path)
    test_X, test_Y = get_X_Y(test_path)
    train_dataset = CreateData(train_X, train_Y, translate_to_id)
    valid_dataset = CreateData(valid_X, valid_Y, translate_to_id)
    test_dataset = CreateData(test_X, test_Y, translate_to_id)

    model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE)

    for i in range(10):
        X = train_dataset[i]["id_text"]
        #unsqueeze(0): 次元が増える, forwardの引数が3次元じゃないとダメっぽい
        #nn.module内で関数的に呼び出されると__call__によってforwardが実行されるようになってるっぽい
        #model(X.unsqueeze(0)).size() = ([1, 4]) dim=-1で次元4の方についてsoftmaxを適用
        print(torch.softmax(model(X.unsqueeze(0)), dim=-1))

"""
tensor([[0.1665, 0.1250, 0.4486, 0.2599]], grad_fn=<SoftmaxBackward>)
tensor([[0.1836, 0.2506, 0.2522, 0.3136]], grad_fn=<SoftmaxBackward>)
tensor([[0.2962, 0.0765, 0.0962, 0.5312]], grad_fn=<SoftmaxBackward>)
tensor([[0.1284, 0.3798, 0.3805, 0.1113]], grad_fn=<SoftmaxBackward>)
tensor([[0.3384, 0.1948, 0.2674, 0.1993]], grad_fn=<SoftmaxBackward>)
tensor([[0.1711, 0.1249, 0.1071, 0.5969]], grad_fn=<SoftmaxBackward>)
tensor([[0.4289, 0.2573, 0.1126, 0.2012]], grad_fn=<SoftmaxBackward>)
tensor([[0.2803, 0.3621, 0.1414, 0.2162]], grad_fn=<SoftmaxBackward>)
tensor([[0.2628, 0.2221, 0.2491, 0.2659]], grad_fn=<SoftmaxBackward>)
tensor([[0.4315, 0.1208, 0.1861, 0.2617]], grad_fn=<SoftmaxBackward>)
"""