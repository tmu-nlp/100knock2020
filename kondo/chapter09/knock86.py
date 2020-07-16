"""
86. 畳み込みニューラルネットワーク (CNN)Permalink
ID番号で表現された単語列x=(x1,x2,…,xT)がある．
ただし，Tは単語列の長さ，xt∈ℝVは単語のID番号のone-hot表記である（Vは単語の総数である）．
畳み込みニューラルネットワーク（CNN: Convolutional Neural Network）を用い，単語列xからカテゴリyを予測するモデルを実装せよ．

ただし，畳み込みニューラルネットワークの構成は以下の通りとする．

単語埋め込みの次元数: dw
畳み込みのフィルターのサイズ: 3 トークン
畳み込みのストライド: 1 トークン
畳み込みのパディング: あり
畳み込み演算後の各時刻のベクトルの次元数: dh
畳み込み演算後に最大値プーリング（max pooling）を適用し，入力文をdh次元の隠れベクトルで表現

すなわち，時刻tの特徴ベクトルpt∈ℝdhは次式で表される．

pt=g(W(px)[emb(xt−1);emb(xt);emb(xt+1)]+b(p))

ただし，W(px)∈ℝdh×3dw,b(p)∈ℝdhはCNNのパラメータ，
gは活性化関数（例えばtanhやReLUなど），
[a;b;c]はベクトルa,b,cの連結である．
なお，行列W(px)の列数が3dwになるのは，3個のトークンの単語埋め込みを連結したものに対して，線形変換を行うためである．

最大値プーリングでは，特徴ベクトルの次元毎に全時刻における最大値を取り，入力文書の特徴ベクトルc∈ℝdhを求める．c[i]でベクトルcのi番目の次元の値を表すことにすると，最大値プーリングは次式で表される．

c[i]=max1≤t≤Tpt[i]
最後に，入力文書の特徴ベクトルcに行列W(yc)∈ℝL×dhとバイアス項b(y)∈ℝLによる線形変換とソフトマックス関数を適用し，カテゴリyを予測する．

y=softmax(W(yc)c+b(y))
なお，この問題ではモデルの学習を行わず，ランダムに初期化された重み行列でyを計算するだけでよい．
"""

import torch
from torch.nn import functional as F
from torch import nn

from knock80 import translate_to_id
from knock81 import get_X_Y, CreateData
from knock85 import word2vec

train_path = "../chapter06/train.csv"
valid_path = "../chapter06/valid.csv"
test_path = "../chapter06/test.csv"

_, V = translate_to_id("")

VOCAB_SIZE = V+1
EMB_SIZE = 300
PADDING_IDX = V
OUTPUT_SIZE = 4
#入力データを幾つの視点からみるか(今回はword2vecによる表現のみ考えるので1個)
IN_CHANNELS = 1
OUT_CHANNELS = 100
#フィルタのサイズがこれ*emb_size
KERNEL_HEIGHTS = 3
#フィルタの動かし幅
STRIDE = 1
#どれだげパディングするか
PADDING = 1

with open("id_file", encoding="utf-8") as f:
    ids = {}
    for line in f:
        id, word = line.split()
        ids[word] = int(id)

class CNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, in_channels, out_channels, kernel_heights, stride, padding, emb_weights=None):
        super().__init__()
        if emb_weights != None:
            self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        #畳み込み計算
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_heights, emb_size), stride, (padding, 0))
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(out_channels, output_size)

    def forward(self, x):
        #print(f"emb.size(): {x.size()}")
        #x→[batch_size, 文長]
        emb = self.emb(x).unsqueeze(1)
        #print(f"emb.size(): {emb.size()}")
        #emb→[batch_size, in_channels, 文長, emb_size]
        conv = self.conv(emb)
        #print(f"conv.size(): {conv.size()}")
        #conv→[batch_size, out_channels, 文長, 1]
        act = F.relu(conv.squeeze(3))
        #print(f"act.size(): {act.size()}")
        #act→[batch_size, out_channels, 文長]
        #文長の最大次元数
        max_pool = F.max_pool1d(act, act.size()[2])
        #print(f"max_pool.size(): {max_pool.size()}")
        #x→[batch_size, 文長]
        out = self.fc(self.drop(max_pool.squeeze(2)))
        return out

if __name__ == "__main__":
    train_X, train_Y = get_X_Y(train_path)
    valid_X, valid_Y = get_X_Y(valid_path)
    test_X, test_Y = get_X_Y(test_path)
    train_dataset = CreateData(train_X, train_Y, translate_to_id)
    valid_dataset = CreateData(valid_X, valid_Y, translate_to_id)
    test_dataset = CreateData(test_X, test_Y, translate_to_id)

    W = word2vec(ids)
    model = CNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, IN_CHANNELS, OUT_CHANNELS, KERNEL_HEIGHTS, STRIDE, PADDING, emb_weights=W)
    for i in range(10):
        X = train_dataset[i]["id_text"]
        print(torch.softmax(model(X.unsqueeze(0)), dim=-1))

"""
tensor([[0.2254, 0.2583, 0.2490, 0.2673]], grad_fn=<SoftmaxBackward>)
tensor([[0.2191, 0.2441, 0.2596, 0.2771]], grad_fn=<SoftmaxBackward>)
tensor([[0.2363, 0.2160, 0.2867, 0.2610]], grad_fn=<SoftmaxBackward>)
tensor([[0.2249, 0.2163, 0.3192, 0.2396]], grad_fn=<SoftmaxBackward>)
tensor([[0.2375, 0.2347, 0.2464, 0.2815]], grad_fn=<SoftmaxBackward>)
tensor([[0.2130, 0.2439, 0.2806, 0.2625]], grad_fn=<SoftmaxBackward>)
tensor([[0.2249, 0.3092, 0.2262, 0.2397]], grad_fn=<SoftmaxBackward>)
tensor([[0.2408, 0.2254, 0.2621, 0.2716]], grad_fn=<SoftmaxBackward>)
tensor([[0.1997, 0.2498, 0.2835, 0.2670]], grad_fn=<SoftmaxBackward>)
tensor([[0.2307, 0.2393, 0.2624, 0.2676]], grad_fn=<SoftmaxBackward>)
"""