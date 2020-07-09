"""
74. 正解率の計測
問題73で求めた行列を用いて学習データおよび評価データの事例を分類したとき，
その正解率をそれぞれ求めよ．
"""
import torch

from knock70 import MyDataset


def calc_accuracy(net, dataset, *, device=None):
    net.eval()
    with torch.no_grad():
        if device:
            preds = torch.argmax(net(dataset.X.to(device)), dim=1).to("cpu")
        else:
            preds = torch.argmax(net(dataset.X), dim=1)
    return (preds == dataset.y).sum().item() / len(dataset)


if __name__ == "__main__":
    net = torch.load("./model/knock73-net.pt")

    train = torch.load("./data/train.pt")
    test = torch.load("./data/test.pt")

    acc_train = calc_accuracy(net, train)
    acc_test = calc_accuracy(net, test)
    print(f"Accuracy (train): {acc_train:f}")
    print(f"Accuracy (test) : {acc_test:f}")


"""result
epoch=10, lr=1e-1
Accuracy (train): 0.280045
Accuracy (test) : 0.284644
"""
