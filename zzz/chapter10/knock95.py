# use en_pretrained_model = 'xlnet-base-cased' in config file
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer
import math


import zzz.chapter10.config as config
from zzz.chapter10.knock90 import get_loader


class TransFormeLate(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout):
        super(TransFormeLate, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def train(train_loader, val_loader, test_loader, epochs=1):
    model = TransFormeLate(config.ntokens,
                           config.emsize,
                           config.nhead,
                           config.nhid,
                           config.nlayers,
                           config.dropout)

    criterion = nn.CrossEntropyLoss()
    lr = 5.0  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    en_tokenizer = BertTokenizer.from_pretrained(config.en_pretrained_model)
    ja_tokenizer = BertTokenizer.from_pretrained(config.ja_pretrained_model)

    model.train()  # Turn on the train mode
    total_loss = 0.
    for _ in range(epochs):
        for en_tokens, ja_tokens, item in tqdm(train_loader):
            ja_tokens = ja_tokens.view(-1)
            optimizer.zero_grad()
            output = model(en_tokens)
            loss = criterion(output.view(-1, config.ntokens), ja_tokens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            # log_interval = 200
        val_loss = evaluate(model, val_loader)
        print('val loss at epoch {0}: {1}'.format(_, val_loss))
    test_loss = evaluate(model, test_loader)
    print('test loss: {0}'.format(test_loss))
    return model


def evaluate(eval_model, loader):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for en_tokens, ja_tokens, item in loader:
            ja_tokens = ja_tokens.view(-1)
            output = eval_model(en_tokens)
            output_flat = output.view(-1, config.ntokens)
            total_loss += len(en_tokens) * criterion(output_flat, ja_tokens).item()
    return total_loss / (len(loader) - 1)

if __name__ == '__main__':
    train_loader = get_loader(train=True)
    val_loader = get_loader(val=True)
    test_loader = get_loader(test=True)
    model = train(train_loader, val_loader, test_loader, epochs=10)
    model.save(model.state_dict(), config.model_path)
