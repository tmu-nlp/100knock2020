# 問題73のコードを改変し，各エポックのパラメータ更新が完了するたびに，
# 訓練データでの損失，正解率，検証データでの損失，正解率をグラフにプロットし，
# 学習の進捗状況を確認できるようにせよ．

import matplotlib.pyplot as plt

net = nn.Linear(300, 4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

train_losses = []
valid_losses = []
train_accs = []
valid_accs = []

for epoc in range(100):
    optimizer.zero_grad()
    y_pred = net(X_train_torch[:4])
    loss = criterion(y_pred, y_train[:4])
    loss.backward()
    optimizer.step()

    train_losses.append(loss)
    valid_losses.append(criterion(net(X_valid_torch), y_valid))

    _, y_pred_train = torch.max(net(X_train_torch[:4]), 1)
    train_accs.append((y_pred_train == y_train[:4]).sum().item() / len(y_train[:4]))
    _, y_pred_valid = torch.max(net(X_valid_torch), 1)
    valid_accs.append((y_pred_valid == y_valid).sum().item() / len(y_valid))

plt.plot(train_losses, label='train loss')
plt.plot(valid_losses, label='valid loss')
plt.legend()
plt.show()

plt.plot(train_accs, label='train acc')
plt.plot(valid_accs, label='valid acc')
plt.legend()
plt.show()