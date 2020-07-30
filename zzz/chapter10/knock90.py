import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import os
import json
import numpy as np
from transformers import BertTokenizer

import zzz.chapter10.config as config


def get_loader(train=False, val=False, test=False, second_train_file=False):
    assert train + val + test == 1
    if train:
        split = 'train'
    elif val:
        split = 'dev'
    else:
        split = 'test'

    if not second_train_file:
        en_file = os.path.join(config.path,
                               config.en_file.format(split))
        ja_file = os.path.join(config.path,
                               config.ja_file.format(split))
    else:
        en_file = config.en_file2
        ja_file = config.ja_file2

    kftt = KFTT(en_file,
                ja_file,
                train,
                val,
                test)

    loader = data.DataLoader(
        kftt,
        batch_size=config.batch_size,
        shuffle=train,  # only shuffle the data in training
        pin_memory=True,
        num_workers=config.data_workers,
    )
    print('Loading {0} data loader finished.'.format(split))
    return loader


class KFTT(data.Dataset):
    def __init__(self, en_file, ja_file, train=False, val=False, test=False):
        assert train + val + test == 1
        super(KFTT, self).__init__()

        with open(en_file, 'r') as f:
            self.en_docs = [line for line in f]
        with open(ja_file, 'r') as f:
            self.ja_docs = [line for line in f]

        self.en_tokenizer = BertTokenizer.from_pretrained(config.en_pretrained_model)
        self.ja_tokenizer = BertTokenizer.from_pretrained(config.ja_pretrained_model)

        print('\nTokenizing English documents.')
        self.en_tokens = [self._encode(doc, self.en_tokenizer) for doc in tqdm(self.en_docs)]
        print('\nTokenizing Japanese documents.')
        self.ja_tokens = [self._encode(doc, self.ja_tokenizer) for doc in tqdm(self.ja_docs)]

    def _encode(self, doc, tokenizer):
        """ Turn a question into a vector of indices and a question length using bert"""

        def seq_padding(X, max_len=config.max_len, padding=0):
            ML = max_len
            res = np.array(
                X + [padding] * (ML - len(X)) if len(X) < ML else X[:ML]
            )
            return res

        res = tokenizer.encode(doc, add_special_tokens=True, max_length=config.max_len, truncation=True)
        res = seq_padding(res)
        return torch.tensor(res)

    def __getitem__(self, item):
        return self.en_tokens[item], self.ja_tokens[item], item

    def __len__(self):
        return len(self.en_tokens)


if __name__ == '__main__':
    train_loader = get_loader(train=True)
    for en_token, ja_token, item in train_loader:
        print(en_token.shape, ja_token.shape)
        print(train_loader.get_vocab_size('en'), train_loader.get_vocab_size('ja'))
        break
