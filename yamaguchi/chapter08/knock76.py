import torch
from torch.utils.data import TensorDataset, DataLoader

from chapter08.knock71 import X_train
from chapter08.knock72 import y_train
from chapter08.knock74 import X_valid, y_valid
from chapter08.knock75 import writer


class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(300, 4),
        )
    def forward(self, X):
        return self.net(X)

model = LogisticRegression()

ds = TensorDataset(X_train, y_train)

# DataLoaderを作成
loader = DataLoader(ds, batch_size=1, shuffle=True)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.net.parameters(), lr=1e-1)

for epoch in range(10):
    for xx, yy in loader:
        y_pred = model(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
      y_pred = model(X_train)
      loss = loss_fn(y_pred, y_train)
      writer.add_scalar('Loss/train', loss, epoch)
      writer.add_scalar('Accuracy/train', accuracy(y_pred,y_train), epoch)

      y_pred = model(X_valid)
      loss = loss_fn(y_pred, y_valid)
      writer.add_scalar('Loss/valid', loss, epoch)
      writer.add_scalar('Accuracy/valid', accuracy(y_pred,y_valid), epoch)

      torch.save(model.state_dict(), 'output/'+str(epoch)+'.model')
      torch.save(optimizer.state_dict(), 'output/'+str(epoch)+'.param')