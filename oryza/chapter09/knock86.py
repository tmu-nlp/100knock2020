from torch.utils.data import DataLoader
from knock81 import CreateDataset
from knock80 import get_feature, word2ids, get_label
import joblib
import time
import torch
from torch import nn
from torch.nn import functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, out_channels, kernel_heights, stride, padding, emb_weights=None):
        super().__init__()
        if emb_weights != None:
            self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.conv = nn.Conv2d(1, out_channels, (kernel_heights, emb_size), stride, (padding, 0))
        self.fc = nn.Linear(out_channels, output_size)

    def forward(self, x):
        emb = self.emb(x).unsqueeze(1)
        conv = self.conv(emb)
        act = F.relu(conv.squeeze(3))
        max_pool = F.max_pool1d(act, act.size()[2])
        out = self.fc(max_pool.squeeze(2))
        return out

if __name__ == "__main__":
    x_train = get_feature(open('train2.feature.txt'))
    y_train = joblib.load('train_label.pkl').tolist()
    x_valid = get_feature(open('valid2.feature.txt'))
    y_valid = joblib.load('valid_label.pkl').tolist()
    x_test = get_feature(open('test2.feature.txt'))
    y_test = joblib.load('test_label.pkl').tolist()

    tr_text2id, tr_word2id = word2ids(x_train)
    train_set = CreateDataset(x_train, y_train, tr_text2id)
    val_text2id, val_word2id = word2ids(x_valid)
    valid_set = CreateDataset(x_valid, y_valid, val_text2id)
    test_text2id, test_word2id = word2ids(x_test)
    test_set = CreateDataset(x_test, y_test, test_text2id)

    VOCAB_SIZE = len(set(tr_word2id.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(tr_word2id.values()))
    OUTPUT_SIZE = 4
    OUT_CHANNELS = 100
    KERNEL_HEIGHTS = 3
    STRIDE = 1
    PADDING = 1

    model = CNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, OUT_CHANNELS, KERNEL_HEIGHTS, STRIDE, PADDING)

    for i in range(10):
        X = train_set[i]['inputs']
        print(torch.softmax(model(X.unsqueeze(0)), dim=-1))