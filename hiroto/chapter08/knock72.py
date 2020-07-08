import torch
from torch import nn
import torch.nn.functional as F
import pickle

with open('./data/train_labels.pickle', mode='rb') as train_l\
        , open('./data/valid_labels.pickle', mode='rb') as valid_l\
        , open('./data/test_labels.pickle', mode='rb') as test_l\
        , open('./data/train_vectors.pickle', mode='rb') as train_v\
        , open('./data/valid_vectors.pickle', mode = 'rb') as valid_v\
        , open('./data/test_vectors.pickle', mode = 'rb') as test_v:
        train_labels = pickle.load(train_l)
        valid_labels = pickle.load(valid_l)
        test_labels = pickle.load(test_l)
        train_vectors = pickle.load(train_v)
        valid_vectors = pickle.load(valid_v)
        test_vectors = pickle.load(test_v)

class Net(nn.Module):
    def __init__(self, in_size, out_size):
        super(Net, self).__init__()
        self.fc = nn.Linear(in_size, out_size, bias=False)
    
    def forward(self, x):
        x = self.fc(x)
        x = F.softmax(x, dim=-1)
        return x

print('####### x1 ##################################')
model = Net(300, 4)
criterion = nn.CrossEntropyLoss()
inputs = train_vectors[:1]
targets = train_labels[:1]
outputs = model(inputs)
loss = criterion(outputs, targets)
model.zero_grad()
loss.backward()
print(f'Loss:\n{loss}')
print(f'grads:\n{model.fc.weight.grad}')

print('####### x1, x2, x3, x4 ##################################')
model = Net(300, 4)
inputs = train_vectors[:4]
targets = train_labels[:4]
outputs = model(inputs)
loss = criterion(outputs, targets)
model.zero_grad()
loss.backward()
print(f'Loss:\n{loss}')
print(f'grads:\n{model.fc.weight.grad}')

'''
####### x1 ##################################
Loss:
1.4022772312164307
grads:
tensor([[-0.0133,  0.0114, -0.0136,  ...,  0.0027, -0.0087,  0.0166],
        [ 0.0042, -0.0037,  0.0044,  ..., -0.0009,  0.0028, -0.0053],
        [ 0.0046, -0.0040,  0.0047,  ..., -0.0009,  0.0030, -0.0058],
        [ 0.0044, -0.0038,  0.0046,  ..., -0.0009,  0.0029, -0.0056]])
####### x1, x2, x3, x4 ##################################
Loss:
1.3799179792404175
grads:
tensor([[-0.0029,  0.0029, -0.0027,  ...,  0.0002, -0.0066,  0.0066],
        [ 0.0028,  0.0005,  0.0004,  ..., -0.0013,  0.0050, -0.0028],
        [-0.0027, -0.0039,  0.0020,  ...,  0.0026, -0.0034, -0.0009],
        [ 0.0028,  0.0005,  0.0004,  ..., -0.0014,  0.0050, -0.0028]])
'''