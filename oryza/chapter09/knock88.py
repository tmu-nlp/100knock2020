from gensim.models import KeyedVectors
from torch.utils.data import DataLoader
from knock81 import CreateDataset
from knock80 import get_feature, word2ids, get_label
from knock82 import calculate_loss_and_accuracy, train_model
from knock83 import Padsequence
from knock84 import RNN
import joblib
import time
import torch
from torch import optim, nn

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
    HIDDEN_SIZE = [64, 128, 256, 512]
    NUM_LAYERS = 2
    LEARNING_RATE = 5e-2 # [5e-2, 5e-3, 5e-4]
    BATCH_SIZE = 32
    NUM_EPOCHS = 10

    # for lr in LEARNING_RATE:
    for h in HIDDEN_SIZE:
        # print(f'Learning Rate: {lr}')
        print(f'Hidden Size: {h}')        
        model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, h, NUM_LAYERS, bidirectional=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

        log = train_model(train_set, valid_set, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX))