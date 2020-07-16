# 別ファイルの変数をインポート
from chapter09.knock80 import words
# 機械学習ライブラリをインポート
import torch

# パラメータを例示されていた通りに適当に設定
dw = 300
dh = 50

# RNNを用いて単語列xからカテゴリyを予測するモデルを実装
class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(len(words)+1,dw)
        self.rnn = torch.nn.RNN(dw,dh,batch_first=True)
        self.linear = torch.nn.Linear(dh,4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x, h=None):
        x = self.emb(x)
        y, h = self.rnn(x, h)
        y = y[:,-1,:]
        y = self.linear(y)
        y = self.softmax(y)
        return y