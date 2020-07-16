from tqdm import tqdm
import torch
from torch import optim
from torchtext import data
from transformers import BertForSequenceClassification

def eval_net(model, data_loader, device='cpu'):
    model.eval()
    ys = []
    ypreds = []
    for (x, y), _ in data_loader:
        with torch.no_grad():
            loss, logit = model(input_ids=x, labels=y) # get lost and gradient
            _, y_pred = torch.max(logit, 1) # obtain predicted labels
            ys.append(y)
            ypreds.append(y_pred)
    ys = torch.cat(ys) # concatenantes list into tensors
    ypreds = torch.cat(ypreds)
    print(f'test acc: {(ys == ypreds).sum().item() / len(ys)}') # calculate the accuracy
    return

# Converts data into tensors
TEXT = data.Field(sequential=True, lower=True, batch_first=True)
LABELS = data.Field(sequential=False, batch_first=True, use_vocab=False)

# Loads dataset from file, put it in variables and use the data converter to tensor from previous step
train, val, test = data.TabularDataset.splits(path='data',train='train2.feature.txt', validation='valid2.feature.txt', test='test2.feature.txt', format='tsv', fields=[('TEXT', TEXT), ('LABEL', LABELS)])

# Device definition if GPU available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Splits data into batches following the batch size
train_iter, val_iter, test_iter = data.Iterator.splits((train, val, test), batch_sizes=(64, 64, 64), device=device, repeat=False, sort=False)

TEXT.build_vocab(train, min_freq=2) # only uses words with frequency => 2
LABELS.build_vocab(train)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4) # Model initialization and load the pre-trained BERT model
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in tqdm(range(10)):
    losses = []
    model.train()
    for batch in train_iter:
        x, y = batch.TEXT, batch.LABEL
        loss, logit = model(input_ids=x, labels=y) 
        model.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        _, y_pred_train = torch.max(logit, 1)
    eval_net(model, test_iter, device)

#   0%|                                                                                                                                                                                | 0/10 [00:00<?, ?it/s]
# test acc: 0.4868913857677903
#  10%|████████████████▍                                                                                                                                                   | 1/10 [30:22<4:33:20, 1822.28s/it]
# test acc: 0.5183520599250936
#  20%|████████████████████████████████▊                                                                                                                                   | 2/10 [57:53<3:56:07, 1770.91s/it]
# test acc: 0.6269662921348315