"""
72. 損失と勾配の計算
(ry
"""
import torch
from torch import nn

from knock70 import MyDataset
from knock71 import SingleLayerNet

if __name__ == "__main__":
    train = torch.load("./data/train.pt")
    d = train.X.shape[1]
    L = len(torch.unique(train.y))
    net = SingleLayerNet(d, L)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(net.forward(train.X[:4]), train.y[:4])
    net.zero_grad()
    loss.backward()

    print("loss:", loss)
    print("grad:", net.fc1.weight.grad)

"""result
loss: tensor(1.8793, grad_fn=<NllLossBackward>)
grad: tensor([[ 0.0155,  0.0013,  0.0011,  ...,  0.0042,  0.0033,  0.0016],
        [ 0.0157,  0.0239, -0.0201,  ..., -0.0070, -0.0088, -0.0160],
        [-0.0270, -0.0297,  0.0330,  ...,  0.0013,  0.0332,  0.0249],
        [-0.0042,  0.0045, -0.0141,  ...,  0.0015, -0.0276, -0.0104]])
"""
