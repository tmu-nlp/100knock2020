import torch
from torch import nn

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc = nn.Linear(300, 4, bias=False)

  def forward(self, x):
    x = self.fc(x)
    return x

net = Net()
y_hat_1 = torch.softmax(net.forward(X_train_torch[:4]), dim=-1)
print(y_hat_1)

"""
tensor([[0.2371, 0.2535, 0.2563, 0.2531],
        [0.2397, 0.2676, 0.2463, 0.2464],
        [0.2507, 0.2495, 0.2480, 0.2518],
        [0.2479, 0.2576, 0.2428, 0.2517]], grad_fn=<SoftmaxBackward>)

"""