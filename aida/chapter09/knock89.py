import numpy as numpy
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import BertModel, BertTokenizer

from knock80 import obtain_data

class CreateDataset(Dataset):
  def __init__(self, X, y, tokenizer, max_len):
    self.X = X
    self.y = y
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.y)

  def __getitem__(self, index): 
    text = self.X[index]
    inputs = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_len,
      pad_to_max_length=True
    )
    ids = inputs['input_ids']
    mask = inputs['attention_mask']

    return {
      'ids': torch.LongTensor(ids),
      'mask': torch.LongTensor(mask),
      'labels': torch.Tensor(self.y[index])
    }

def create_onehot(y):
  Y = torch.zeros(len(y), 4)
  for i, label in enumerate(y):
    Y[i][label] += 1
  return Y

def calculate_loss_and_acc_bert(bert, criterion, loader, device):
  bert.eval()
  loss = 0.0
  total = 0
  correct = 0
  with torch.no_grad():
    for data in loader:
      ids = data['ids'].to(device)
      mask = data['mask'].to(device)
      labels = data['labels'].to(device)

      outputs = bert.forward(ids, mask)
      loss += criterion(outputs, labels).item()
      pred = torch.argmax(outputs, dim=-1).cpu().numpy()
      labels = torch.argmax(labels, dim=-1).cpu().numpy() 
      total += len(labels)
      correct += (pred == labels).sum().item()

  return loss / len(loader), correct / total


def train_bert(dataset_train, dataset_valid, batch_size, bert, criterion, optimizer, num_epochs, device=None):
  bert.to(device)

  dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
  dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)

  log_train = []
  log_valid = []
  for epoch in tqdm(range(num_epochs)):
    bert.train()
    for data in dataloader_train:
      ids = data['ids'].to(device)
      mask = data['mask'].to(device)
      labels = data['labels'].to(device)

      optimizer.zero_grad()
      outputs = bert.forward(ids, mask)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

    loss_train, acc_train = calculate_loss_and_acc_bert(bert, criterion, dataloader_train, device)
    loss_valid, acc_valid = calculate_loss_and_acc_bert(bert, criterion, dataloader_valid, device)
    log_train.append([loss_train, acc_train])
    log_valid.append([loss_valid, acc_valid])

    print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, {(e_time - s_time):.4f}sec') 

  return {'train': log_train, 'valid': log_valid}


if __name__ == '__main__':
  file_train = './../chapter06/train.feature.txt'
  file_dev = './../chapter06/dev.feature.txt'
  file_test = './../chapter06/test.feature.txt'
  X_train, y_train, _ = obtain_data(file_train)
  X_dev, y_dev, _ = obtain_data(file_dev)
  X_test, y_test, _ = obtain_data(file_test)

  max_len = 20
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  dataset_train = CreateDataset(X_train, create_onehot(y_train), tokenizer, max_len)
  dataset_valid = CreateDataset(X_dev, create_onehot(y_dev), tokenizer, max_len)
  dataset_test = CreateDataset(X_test, create_onehot(y_test), tokenizer, max_len)

  DROP_RATE = 0.4
  OUTPUT_SIZE = 4
  BATCH_SIZE = 32
  NUM_EPOCHS = 4
  LEARNING_RATE = 2e-5

  bert = BERTClassifier(DROP_RATE, OUTPUT_SIZE)
  criterion = torch.nn.BCEWithLogitsLoss()
  optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
  device = torch.device('cuda')

  log = train_bert(dataset_train, dataset_dev, BATCH_SIZE, bert, criterion, optimizer, NUM_EPOCHS, device=device)

"""
epoch: 1, loss_train: 0.0859, accuracy_train: 0.9516, loss_dev: 0.1142, accuracy_dev: 0.9229
epoch: 2, loss_train: 0.0448, accuracy_train: 0.9766, loss_dev: 0.1046, accuracy_dev: 0.9259
epoch: 3, loss_train: 0.0316, accuracy_train: 0.9831, loss_dev: 0.1082, accuracy_dev: 0.9266
epoch: 4, loss_train: 0.0170, accuracy_train: 0.9932, loss_dev: 0.1179, accuracy_dev: 0.9289
"""

