# 問題75のコードを改変し，
# 各エポックのパラメータ更新が完了するたびに，
# チェックポイント（学習途中のパラメータ（重み行列など）の値や最適化アルゴリズムの内部状態）
# をファイルに書き出せ．

net = nn.Linear(300, 4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

for epoc in range(100):
    optimizer.zero_grad()
    y_pred = net(X_train_torch[:4])
    loss = criterion(y_pred, y_train[:4])
    loss.backward()
    optimizer.step()

    train_losses.append(loss)
    valid_losses.append(loss_fn(net(X_valid_torch), y_valid))
    
    torch.save(net.state_dict(), 'checkpoint' + str(epoc))
    
