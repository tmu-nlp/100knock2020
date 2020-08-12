'''
unfinished
'''
from tensorboardX import SummaryWriter
import torch
from transformers import BertTokenizer

from zzz.chapter10 import config
from zzz.chapter10.knock90 import get_loader
from zzz.chapter10.knock92 import run
from zzz.chapter10.knock93 import eval

writer = SummaryWriter('./Result')

if __name__ == '__main__':
    model = torch.load(config.model_path)
    ja_tokenizer = BertTokenizer.from_pretrained(config.ja_pretrained_model)
    en_tokenizer = BertTokenizer.from_pretrained(config.en_pretrained_model)
