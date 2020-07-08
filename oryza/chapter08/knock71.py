import joblib
from torch import nn
import torch

class NN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.h1 = nn.Linear(input_size,hidden_size)
    self.output = nn.Linear(hidden_size, output_size)
    nn.init.normal_(self.h1.weight, 0.0, 1.0)
    nn.init.normal_(self.output.weight, 0.0, 1.0)

  def forward(self, x):
    x = self.h1(x)
    x = self.output(x)
    return x


if __name__ == "__main__":
    trainf_np = joblib.load('train_feature.pkl')
    train_x = torch.from_numpy(trainf_np)

    model = NN(300, 256, 4) 
    prediction = torch.softmax(model.forward(train_x[:4].float()), dim=-1)
    print(prediction)

'''
tensor([[0.2868, 0.0627, 0.1437, 0.5068],
        [0.2474, 0.0813, 0.1797, 0.4916],
        [0.2305, 0.0404, 0.1888, 0.5403],
        [0.3143, 0.0564, 0.1772, 0.4521]], grad_fn=<SoftmaxBackward>)
'''