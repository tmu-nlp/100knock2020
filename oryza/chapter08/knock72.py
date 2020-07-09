import joblib
from torch import nn
import torch
from knock71 import NN

if __name__ == "__main__":
    trainf_np = joblib.load('train_feature.pkl')
    train_x = torch.from_numpy(trainf_np)
    trainl_np = joblib.load('train_label.pkl')
    train_y = torch.from_numpy(trainl_np)

    model = NN(300, 256, 4) 
    criterion = nn.CrossEntropyLoss()

    loss = criterion(model(train_x[:4].float()), train_y[:4])
    model.zero_grad()
    loss.backward()

    print('loss: ' + str(loss))
    print('gradient: \n' + str(model.output.weight.grad))

'''
loss: tensor(2.9821, grad_fn=<NllLossBackward>)
gradient:
tensor([[-0.9348,  0.3831,  1.1587,  ...,  0.5627, -0.2966,  0.1505],
        [ 3.6555, -1.5159, -4.5666,  ..., -2.2121,  1.1609, -0.6112],
        [-1.0458,  0.4359,  1.3220,  ...,  0.6385, -0.3333,  0.1830],
        [-1.6749,  0.6969,  2.0859,  ...,  1.0108, -0.5310,  0.2777]])
'''