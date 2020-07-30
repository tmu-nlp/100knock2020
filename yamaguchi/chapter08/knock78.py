import torch
import time
from torch.utils.data import TensorDataset, DataLoader
from chapter08.knock71 import X_train
from chapter08.knock72 import y_train

class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(300, 4),
        )
    def forward(self, X):
        return self.net(X)

model = LogisticRegression()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

ds = TensorDataset(X_train.to(device), y_train.to(device))
loss_fn = torch.nn.CrossEntropyLoss()

ls_bs = [2**i for i in range(15)]
ls_time = []

for bs in ls_bs:
  loader = DataLoader(ds, batch_size=bs, shuffle=True)
  optimizer = torch.optim.SGD(model.net.parameters(), lr=1e-1)
  for epoch in range(1):
      start = time.time()
      for xx, yy in loader:
          y_pred = model(xx)
          loss = loss_fn(y_pred, yy)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
  ls_time.append(time.time()-start)

print (ls_time)