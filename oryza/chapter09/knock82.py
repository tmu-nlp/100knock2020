from torch.utils.data import DataLoader
from knock81 import CreateDataset, RNN
from knock80 import get_feature, word2ids, get_label
import joblib
import time
import torch
from torch import optim, nn

def calculate_loss_and_accuracy(model, dataset, device=None, criterion=None):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for data in dataloader:
            inputs = data['inputs'].to(device)
            labels = data['labels'].to(device)

            outputs = model(inputs)

            if criterion != None:
                loss += criterion(outputs, labels).item()

            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()

    return loss / len(dataset), correct / total


def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, collate_fn=None, device=None):
    model.to(device)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False)

    log_train = []
    log_valid = []
    for epoch in range(num_epochs):
        s_time = time.time()

        model.train()
        for data in dataloader_train:
            optimizer.zero_grad()

            inputs = data['inputs'].to(device)
            labels = data['labels'].to(device)
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()

        loss_train, acc_train = calculate_loss_and_accuracy(model, dataset_train, device, criterion=criterion)
        loss_valid, acc_valid = calculate_loss_and_accuracy(model, dataset_valid, device, criterion=criterion)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'checkpoint{epoch + 1}.pt')

        e_time = time.time()

        print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, {(e_time - s_time):.4f}sec') 

    return {'train': log_train, 'valid': log_valid}

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
    HIDDEN_SIZE = 50
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 1
    NUM_EPOCHS = 10

    model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    log = train_model(train_set, valid_set, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS)