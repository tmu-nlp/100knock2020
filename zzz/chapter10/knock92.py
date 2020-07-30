import torch
from transformers import BertTokenizer

from zzz.chapter10 import config

docs = [
    'A transformer model. User is able to modify the attributes as needed. The architecture is based on the paper “Attention Is All You Need”.',
    'Thank you for your response. Since a lot of us trying to use transformers in production too, please consider having stable workflow.',
    'It is useful when training a classification problem with C classes.']


def run(model, ja_tokenizer, en_tokenizer):
    with torch.no_grad():
        for doc in docs:
            en_token = en_tokenizer.encode(doc, add_special_tokens=True, max_length=config.max_len, truncation=True)
            output = model(en_token)
            print(ja_tokenizer.decode(output.view(-1)))

if __name__ == '__main__':
    model = torch.load(config.model_path)
    ja_tokenizer = BertTokenizer.from_pretrained(config.ja_pretrained_model)
    en_tokenizer = BertTokenizer.from_pretrained(config.en_pretrained_model)
    run(model, ja_tokenizer, en_tokenizer)
