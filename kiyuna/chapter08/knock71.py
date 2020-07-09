"""
71. 単層ニューラルネットワークによる予測
(ry
"""
import torch
from torch import nn

from knock70 import MyDataset


class SingleLayerNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        # 重みの初期化
        # 正規分布 weight, mean, std
        nn.init.normal_(self.fc1.weight, 0.0, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        return x


if __name__ == "__main__":
    train = torch.load("./data/train.pt")
    d = train.X.shape[1]
    L = len(torch.unique(train.y))
    net = SingleLayerNet(d, L)

    y_hat_1 = torch.softmax(net.forward(train.X[0]), dim=-1)
    print(y_hat_1)

    Y_hat = torch.softmax(net.forward(train.X[:4]), dim=-1)
    print(Y_hat)


"""result
tensor([0.3553, 0.0737, 0.2363, 0.3347], grad_fn=<SoftmaxBackward>)
tensor([[0.3553, 0.0737, 0.2363, 0.3347],
        [0.2200, 0.1276, 0.0506, 0.6017],
        [0.2161, 0.5921, 0.0618, 0.1299],
        [0.5187, 0.1764, 0.3007, 0.0041]], grad_fn=<SoftmaxBackward>)
"""
