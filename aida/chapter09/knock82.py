import pickle
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from knock80 import obtain_data, tokenize
from knock81 import CreateDataset, RNN

def calculate_loss_and_acc(rnn, dataset, criterion, device=None):
  """ compute loss and accuracy
  
  :param rnn: rnn model
  :param dataset: torch.utils.data.Dataset
  :param criterion: compute loss
  :param device: use gpu
  """
  dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
  loss = 0.0
  total = 0
  correct = 0
  with torch.no_grad():
    for data in dataloader:
      inputs = data['inputs'].to(device)
      labels = data['labels'].to(device)

      outputs = rnn(inputs)

      loss += criterion(outputs, labels).item()

      pred = torch.argmax(outputs, dim=-1)
      total += len(inputs)
      correct += (pred == labels).sum().item()

  return loss / len(dataset), correct / total

def train_model(dataset_train, dataset_dev, batch_size, rnn, criterion, optimizer, num_epochs, collate_fn=None, device=None):
  """ train model
  
  :param batch_size: minibatch size of train data
  :param num_epochs: epoch
  :param collate_fn: padding method
  """
  rnn.to(device)

  dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
  dataloader_dev = DataLoader(dataset_dev, batch_size=1, shuffle=False)

  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-5, last_epoch=-1)

  log_train = []
  log_dev = []
  for epoch in range(num_epochs):
    rnn.train()
    for data in dataloader_train:
      optimizer.zero_grad()
      inputs = data['inputs'].to(device)
      labels = data['labels'].to(device)
      outputs = rnn.forward(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

    rnn.eval()
    loss_train, acc_train = calculate_loss_and_acc(rnn, dataset_train, criterion, device)
    loss_dev, acc_dev = calculate_loss_and_acc(rnn, dataset_dev, criterion, device)
    log_train.append([loss_train, acc_train])
    log_dev.append([loss_dev, acc_dev])

    torch.save({'epoch': epoch, 'model_state_dict': rnn.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'checkpoint{epoch + 1}.pt')
    print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_: {loss_dev:.4f}, accuracy_dev: {acc_dev:.4f}') 

    # early stopping
    if epoch > 2 and log_dev[epoch - 3][0] <= log_dev[epoch - 2][0] <= log_dev[epoch - 1][0] <= log_dev[epoch][0]:
      break

    scheduler.step()

  return {'train': log_train, 'dev': log_dev}

if __name__ == '__main__':
  dataset_train = torch.load('./dataset_train.pt')
  dataset_dev = torch.load('./dataset_dev.pt')
  word_to_id = pickle.load(open('./word_to_id.pkl', 'rb'))

  VOCAB_SIZE = len(set(word_to_id.values())) + 1 
  EMB_SIZE = 300
  PADDING_IDX = len(set(word_to_id.values()))
  OUTPUT_SIZE = 4
  HIDDEN_SIZE = 50
  LEARNING_RATE = 1e-3
  BATCH_SIZE = 1
  NUM_EPOCHS = 10

  rnn = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE)
  #device = torch.device('cuda')
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(rnn.parameters(), lr=LEARNING_RATE)

  log = train_model(dataset_train, dataset_dev, BATCH_SIZE, rnn, criterion, optimizer, NUM_EPOCHS)

"""
epoch: 1, loss_train: 1.0727, accuracy_train: 0.5852, loss_dev: 1.0974, accuracy_dev: 0.5704
epoch: 2, loss_train: 1.0230, accuracy_train: 0.6154, loss_dev: 1.0718, accuracy_dev: 0.5884
epoch: 3, loss_train: 0.9756, accuracy_train: 0.6334, loss_dev: 1.0378, accuracy_dev: 0.5947
epoch: 4, loss_train: 0.9364, accuracy_train: 0.6568, loss_dev: 1.0240, accuracy_dev: 0.6063
epoch: 5, loss_train: 0.9022, accuracy_train: 0.6703, loss_dev: 0.9946, accuracy_dev: 0.6265
epoch: 6, loss_train: 0.8540, accuracy_train: 0.6840, loss_dev: 0.9656, accuracy_dev: 0.6444
epoch: 7, loss_train: 0.8207, accuracy_train: 0.7000, loss_dev: 0.9422, accuracy_dev: 0.6624
epoch: 8, loss_train: 0.7946, accuracy_train: 0.7082, loss_dev: 0.9196, accuracy_dev: 0.6783
epoch: 9, loss_train: 0.7809, accuracy_train: 0.7123, loss_dev: 0.9069, accuracy_dev: 0.6783
epoch: 10, loss_train: 0.7765, accuracy_train: 0.7138, loss_dev: 0.9027, accuracy_dev: 0.6751
"""
