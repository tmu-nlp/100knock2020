"""
81. RNNによる予測
(ry

[Ref]
- https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial
"""
import torch
import torch.nn as nn

from knock80 import MyDataset, get_V

d_w = 300
d_h = 50
V = get_V()
L = 4


class RNN(nn.Module):
    def __init__(self, d_w, d_h, L, emb, num_layers=1, bidirectional=False, nonlinearity="tanh"):
        super().__init__()
        self.d_h = d_h
        self.num = num_layers * (1 + bidirectional)
        self.emb = nn.Parameter(emb)
        self.rnn = nn.RNN(
            input_size=d_w,
            hidden_size=d_h,
            num_layers=num_layers,
            bidirectional=bidirectional,
            nonlinearity=nonlinearity,
        )
        self.fc = nn.Linear(d_h * (1 + bidirectional), L)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        emb = x @ self.emb.T
        hidden = self.init_hidden(x.shape[1]).to(x.device)
        output, hidden = self.rnn(emb, hidden)
        return self.softmax(self.fc(output[-1]))

    def init_hidden(self, batch_size):
        return torch.zeros(self.num, batch_size, self.d_h)

    def show_params(self):
        print("-" * 63)
        for param_name, param in self.named_parameters():
            print(f"{param_name:23}", param.shape)
        print("-" * 63)


if __name__ == "__main__":
    train = torch.load("./data/train.pt")
    emb = torch.Tensor(d_w, V).normal_()
    rnn = RNN(d_w, d_h, L, emb)
    rnn.show_params()
    output = rnn(train.X[0].unsqueeze(1))
    print(output, output.shape, sep=", ")


"""result
---------------------------------------------------------------
emb                     torch.Size([300, 7403])
rnn.weight_ih_l0        torch.Size([50, 300])
rnn.weight_hh_l0        torch.Size([50, 50])
rnn.bias_ih_l0          torch.Size([50])
rnn.bias_hh_l0          torch.Size([50])
fc.weight               torch.Size([4, 50])
fc.bias                 torch.Size([4])
---------------------------------------------------------------
tensor([[0.1611, 0.1920, 0.4324, 0.2146]], grad_fn=<SoftmaxBackward>), torch.Size([1, 4])
"""
