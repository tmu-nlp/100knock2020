"""
学習データの事例x1と事例集合x1,x2,x3,x4に対して，クロスエントロピー損失と，行列Wに対する勾配を計算せよ．
なお，ある事例xiに対して損失は次式で計算される．
li=−log[事例xiがyiに分類される確率]
ただし，事例集合に対するクロスエントロピー損失は，その集合に含まれる各事例の損失の平均とする．
"""

import torch
import torch.nn as nn

y_train = torch.from_numpy(y_train.astype(np.int64)).clone()

net = nn.Linear(300, 4)
criterion = nn.CrossEntropyLoss()

l = criterion(net.forward(X_train[:4]), y_train[:4])
net.zero_grad()
l.backward()

print("Loss: " + str(l))
print("Gradient: " + "\n" + str(net.weight.grad))

"""
Loss: tensor(1.3999, grad_fn=<NllLossBackward>)
Gradient:
tensor([[ 0.0184,  0.0135,  0.0021,  ..., -0.0036,  0.0086,  0.0010],
        [ 0.0187,  0.0138,  0.0019,  ..., -0.0038,  0.0086,  0.0017],
        [-0.0219, -0.0173,  0.0175,  ...,  0.0144, -0.0195, -0.0204],
        [-0.0152, -0.0100, -0.0215,  ..., -0.0070,  0.0022,  0.0177]])
"""