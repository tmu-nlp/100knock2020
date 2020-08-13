from transformers import BertTokenizer
import torch

from zzz.chapter10 import config
from zzz.chapter10.knock90 import get_loader
from zzz.chapter10.knock92 import run
from zzz.chapter10.knock93 import eval



if __name__ == '__main__':
    model = torch.load(config.model_path)
    ja_tokenizer = BertTokenizer.from_pretrained(config.ja_pretrained_model)
    en_tokenizer = BertTokenizer.from_pretrained(config.en_pretrained_model)
    run(model, ja_tokenizer, en_tokenizer)
    bleu = eval(model)
    print('=' * 20)
    print('BLEU score: {0}.'.format(bleu))
    print(
    'max_len' , config.max_len,
    'batch_size' , config.batch_size,
    'embedding_size' , config.embedding_size,
    'drop_prob' , config.drop_prob,
    'emsize' , config.emsize,  # embedding dimension
    'nhid' , config.nhid,  # the dimension of the feedforward network model in nn.TransformerEncoder
    'nlayers' , config.nlayers,  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    'nhead' , config.nhead,  # the number of heads in the multiheadattention models
    'dropout' , config.dropout,  # the dropout value
    )
