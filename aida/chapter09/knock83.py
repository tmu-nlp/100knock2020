import pickle
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from knock80 import obtain_data, tokenize
from knock81 import CreateDataset, RNN
from knock82 import calculate_loss_and_acc, train_model

class Padsequence():
  """padding sequence in each minibatch"""
  def __init__(self, padding_idx):
    self.padding_idx = padding_idx

  def __call__(self, batch):
    sorted_batch = sorted(batch, key=lambda x: x['inputs'].shape[0], reverse=True)
    sequences = [x['inputs'] for x in sorted_batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.padding_idx)
    labels = torch.LongTensor([x['labels'] for x in sorted_batch])

    return {'inputs': sequences_padded, 'labels': labels}

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
  BATCH_SIZE = 32 
  NUM_EPOCHS = 10

  rnn = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(rnn.parameters(), lr=LEARNING_RATE)

  log = train_model(dataset_train, dataset_dev, BATCH_SIZE, rnn, criterion, 
                    optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX), device=device)

"""
epoch: 1, loss_train: 1.0687, accuracy_train: 0.6129, loss_dev: 1.1163, accuracy_dev: 0.5926
epoch: 2, loss_train: 1.0061, accuracy_train: 0.6321, loss_dev: 1.0401, accuracy_dev: 0.6053
epoch: 3, loss_train: 0.9022, accuracy_train: 0.6654, loss_dev: 0.9739, accuracy_dev: 0.6148
epoch: 4, loss_train: 0.8375, accuracy_train: 0.6977, loss_dev: 0.9593, accuracy_dev: 0.6370
epoch: 5, loss_train: 0.8625, accuracy_train: 0.6774, loss_dev: 0.9510, accuracy_dev: 0.6233
epoch: 6, loss_train: 0.7032, accuracy_train: 0.7437, loss_dev: 0.8379, accuracy_dev: 0.6794
epoch: 7, loss_train: 0.6693, accuracy_train: 0.7564, loss_dev: 0.8294, accuracy_dev: 0.6815
epoch: 8, loss_train: 0.5748, accuracy_train: 0.7857, loss_dev: 0.7458, accuracy_dev: 0.7090
epoch: 9, loss_train: 0.5543, accuracy_train: 0.7920, loss_dev: 0.7401, accuracy_dev: 0.7026
epoch: 10, loss_train: 0.5499, accuracy_train: 0.7921, loss_dev: 0.7409, accuracy_dev: 0.7037
"""