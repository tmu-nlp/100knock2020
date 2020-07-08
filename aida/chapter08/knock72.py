import torch

from knock71 import SingleLayerPerceptron

X_train = torch.load('./tensors/X_train')
Y_train = torch.load('./tensors/Y_train')

model = SingleLayerPerceptron(X_train.shape[1], 4)

criterion = torch.nn.CrossEntropyLoss()

l_1 = criterion(model.forward(X_train[1]).reshape(-1, 4), Y_train[1].long())
model.zero_grad()
l_1.backward()
print(f'loss: {l_1:.4f}')
print(f'grad:\n{model.fc.weight.grad}')

l = criterion(model.forward(X_train[:4]).reshape(-1, 4), Y_train[:4].reshape(-1).long())
model.zero_grad()
l.backward()
print(f'loss: {l:.4f}')
print(f'grad:\n{model.fc.weight.grad}')

"""
loss: 0.3653
grad:
tensor([[ 1.1469e-03,  8.0004e-04, -1.3207e-03,  ...,  9.5547e-05,
          4.7012e-04, -5.7254e-04],
        [-2.4889e-02, -1.7361e-02,  2.8659e-02,  ..., -2.0734e-03,
         -1.0202e-02,  1.2424e-02],
        [ 2.1754e-02,  1.5174e-02, -2.5050e-02,  ...,  1.8122e-03,
          8.9168e-03, -1.0859e-02],
        [ 1.9881e-03,  1.3868e-03, -2.2893e-03,  ...,  1.6562e-04,
          8.1490e-04, -9.9244e-04]])
loss: 1.4934
grad:
tensor([[ 0.0201, -0.0286, -0.0008,  ..., -0.0187, -0.0213,  0.0052],
        [-0.0188,  0.0052,  0.0075,  ...,  0.0111,  0.0006,  0.0063],
        [ 0.0106,  0.0158, -0.0138,  ...,  0.0102,  0.0164, -0.0128],
        [-0.0118,  0.0076,  0.0071,  ..., -0.0027,  0.0044,  0.0013]])
"""

