import torch
import numpy as np
from chapter08.knock71 import X_train, W, softmax

y_train = np.loadtxt('y_train.txt')
y_train = torch.tensor(y_train, dtype=torch.int64)

loss = torch.nn.CrossEntropyLoss()

print (loss(torch.matmul(X_train[:1], W),y_train[:1]))
print (loss(torch.matmul(X_train[:4], W),y_train[:4]))

ans = []
for s,i in zip(softmax(torch.matmul(X_train[:4], W)),y_train[:4]):
  ans.append(-np.log(s[i]))
print (np.mean(ans))
