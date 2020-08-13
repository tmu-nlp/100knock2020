"""
89. 事前学習済み言語モデルからの転移学習
事前学習済み言語モデル（例えばBERTなど）を出発点として，
ニュース記事見出しをカテゴリに分類するモデルを構築せよ．

[Ref]
- https://github.com/upura/nlp100v2020/blob/master/ch09/ans89.py

[Usage]
python knock89.py | tee knock89.log
"""
# split_names = ["train", "valid", "test"]
# for split_name in split_names:
#     with open(f'./data/{split_name}.feature.txt') as f_in, \
#             open(f'./data/{split_name}.feature.csv', 'w') as f_out:
#         for line in f_in:
#             label, text = line.split('\t')
#             label = "btem".index(label)
#             f_out.write(f'{label}\t{text}')
# exit(1)

import torch
from torch import optim
from tqdm import tqdm

from torchtext import data
from transformers import BertForSequenceClassification


def eval_net(model, data_loader, split_name, device):
    model.eval()
    ys = []
    ypreds = []
    for batch in data_loader:
        with torch.no_grad():
            x, y = batch.TEXT, batch.LABEL
            loss, logit = model(input_ids=x, labels=y)
            _, y_pred = torch.max(logit, 1)
            ys.append(y)
            ypreds.append(y_pred)
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    print(f"{split_name} acc: {(ys == ypreds).sum().item() / len(ys)}")


TEXT = data.Field(sequential=True, lower=True, batch_first=True)
LABELS = data.Field(sequential=False, batch_first=True, use_vocab=False)

train, val, test = data.TabularDataset.splits(
    path="data",
    train="train.feature.csv",
    validation="valid.feature.csv",
    test="test.feature.csv",
    format="tsv",
    fields=[("LABEL", LABELS), ("TEXT", TEXT)],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
train_iter, val_iter, test_iter = data.Iterator.splits(
    (train, val, test), batch_sizes=(64, 64, 64), device=device, repeat=False, sort=False
)

TEXT.build_vocab(train, min_freq=2)
LABELS.build_vocab(train)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in tqdm(range(10)):
    print(f"epoch: {epoch}")
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
    print("loss :", sum(losses) / len(losses))
    eval_net(model, train_iter, "train", device)
    eval_net(model, val_iter, "valid", device)
eval_net(model, test_iter, "test ", device)
