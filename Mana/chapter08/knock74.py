# 問題73で求めた行列を用いて学習データおよび評価データの事例を分類したとき，
# その正解率をそれぞれ求めよ．


y_valid = torch.from_numpy(y_valid.astype(np.int64)).clone()

_, y_pred_train = torch.max(net(X_train_torch[:4]), 1)
print((y_pred_train == y_train[:4]).sum().item() / len(y_train[:4]))

_, y_pred_test = torch.max(net(X_valid_torch), 1)
print((y_pred_test == y_valid).sum().item() / len(y_valid))

# 1.0
# 0.19183673469387755
# ミニバッチにしていないので低い？