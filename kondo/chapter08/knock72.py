"""
72. 損失と勾配の計算Permalink
学習データの事例x1と事例集合x1,x2,x3,x4に対して，クロスエントロピー損失と，行列Wに対する勾配を計算せよ．
なお，ある事例xiに対して損失は次式で計算される．

li=−log[事例xiがyiに分類される確率]
ただし，事例集合に対するクロスエントロピー損失は，その集合に含まれる各事例の損失の平均とする．
"""

import torch
import numpy as np

X_train_data = "X_train.npy"
Y_train_data = "Y_train.npy"
X_valid_data = "X_valid.npy"
Y_valid_data = "Y_valid.npy"
X_test_data = "X_test.npy"
Y_test_data = "Y_test.npy"

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

if __name__ == "__main__":
    #単層ニューラルネットワーク
    model = SLPNet(d, L)
    X = np.load(file=X_train_data)
    X = torch.tensor(X, dtype=torch.float32)
    Y = np.load(file=Y_train_data)  #正解ラベル
    Y = torch.tensor(Y, dtype=torch.int64)
    x1 = X[:1]
    y1 = Y[:1]
    X_1_4 = X[:4]
    Y_1_4 = Y[:4]

    model = SLPNet(300, 4)

    criterion = torch.nn.CrossEntropyLoss()
    loss1 = criterion(model.forward(x1), y1)
    model.zero_grad()
    loss1.backward() #勾配
    print("x1 loss")
    print(loss1)
    print("x1 grad")
    print(model.fc.weight.grad)

    loss2 = criterion(model.forward(X_1_4), Y_1_4)
    model.zero_grad()
    loss2.backward()
    print("x1-4 loss")
    print(loss2)
    print("x1-4 grad")
    print(model.fc.weight.grad)


"""
x1 loss
tensor(2.5288, grad_fn=<NllLossBackward>)
x1 grad
tensor([[-0.0448,  0.0339,  0.0194,  ..., -0.0176, -0.0413,  0.0649],
        [ 0.0149, -0.0113, -0.0065,  ...,  0.0058,  0.0137, -0.0215],
        [ 0.0076, -0.0058, -0.0033,  ...,  0.0030,  0.0071, -0.0111],
        [ 0.0223, -0.0169, -0.0097,  ...,  0.0087,  0.0206, -0.0323]])
x1-4 loss
tensor(2.0938, grad_fn=<NllLossBackward>)
x1-4 grad
tensor([[-0.0008,  0.0072,  0.0154,  ..., -0.0167, -0.0109,  0.0181],
        [-0.0048,  0.0025,  0.0162,  ...,  0.0105,  0.0252, -0.0044],
        [ 0.0006, -0.0043, -0.0338,  ...,  0.0043, -0.0040, -0.0078],
        [ 0.0050, -0.0054,  0.0022,  ...,  0.0019, -0.0103, -0.0059]])
"""










































































"""
import numpy as np
import torch

X_train_data = "X_train.npy"
Y_train_data = "Y_train.npy"
X_valid_data = "X_valid.npy"
Y_valid_data = "Y_valid.npy"
X_test_data = "X_test.npy"
Y_test_data = "Y_test.npy"

d = 300
L = 4

def softmax(x):
    u = np.sum(np.exp(x))
    return np.exp(x)/u

def probably(X, w):
    p = []
    for x in X:
        p.append(softmax(np.dot(x, w)))
    return np.array(p)

def cross_entropy_loss(X, Y):
    u = 0
    for x, y in zip(X, Y):
        u += np.sum(-np.log(x[int(y)]))
    return u/len(Y)

if __name__ == "__main__":
    np.random.seed(7)
    W = np.random.rand(d, L)
    X = np.load(file=X_train_data)
    Y = np.load(file=Y_train_data)  #正解ラベル
    x1 = X[:1]
    y1 = Y[:1]
    X_1_4 = X[:4]
    Y_1_4 = Y[:4]
"""

"""
    y_hat = probably(x1, W)
    Y_hat = probably(X_1_4, W)

    print("x1 cross entropy loss")
    print(cross_entropy_loss(y_hat, y1))
    print("X1-4 cross entropy loss")
    print(cross_entropy_loss(Y_hat, Y_1_4))
"""
