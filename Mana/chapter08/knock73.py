# 確率的勾配降下法（SGD: Stochastic Gradient Descent）を用いて，行列Wを学習せよ．
# なお，学習は適当な基準で終了させればよい（例えば「100エポックで終了」など）．

from torch import optim

net = nn.Linear(X_train_torch.size()[1], 4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

losses = []

for epoc in range(100):
    optimizer.zero_grad()
    y_pred = net(X_train_torch[:4])
    loss = criterion(y_pred, y_train[:4])
    loss.backward()
    optimizer.step()
    losses.append(loss)

print(net.state_dict()['weight'])

"""
tensor([[-0.0557, -0.0489,  0.0335,  ...,  0.0516, -0.0410,  0.0083],
        [-0.0406, -0.0744, -0.0468,  ..., -0.0194,  0.0187, -0.0587],
        [ 0.0060,  0.0613, -0.1088,  ..., -0.0628,  0.1075,  0.1731],
        [ 0.0230,  0.0524,  0.1129,  ..., -0.0113, -0.0700, -0.0826]])
"""