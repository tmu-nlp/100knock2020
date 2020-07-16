from torch.utils.data import DataLoader
from knock81 import CreateDataset
from knock80 import get_feature, word2ids, get_label
from knock82 import calculate_loss_and_accuracy, train_model
from knock83 import Padsequence
from knock86 import CNN
import joblib
import time
import torch
from torch import optim, nn
from torch.nn import functional as F

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
    LEARNING_RATE = 5e-2
    BATCH_SIZE = 64
    NUM_EPOCHS = 10

    model = CNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, OUT_CHANNELS, KERNEL_HEIGHTS, STRIDE, PADDING)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    log = train_model(train_set, valid_set, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX))