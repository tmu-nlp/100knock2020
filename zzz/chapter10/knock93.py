import torch
from torchtext.data.metrics import bleu_score
from transformers import BertTokenizer

from zzz.chapter10 import config
from zzz.chapter10.knock90 import get_loader


def eval(eval_model, data_loader):
    ja_tokenizer = BertTokenizer.from_pretrained(config.ja_pretrained_model)
    en_tokenizer = BertTokenizer.from_pretrained(config.en_pretrained_model)
    candidate_corpus = []
    reference_corpus = []
    with torch.no_grad():
        for en_tokens, ja_tokens, item in data_loader:
            output = eval_model(en_tokens)

            candidate_corpus = candidate_corpus + [
                ja_tokenizer.decode(out, add_special_tokens=True, max_length=config.max_len, truncation=True) for out in
                output]
            reference_corpus = reference_corpus + [
                ja_tokenizer.decode(ja_token, add_special_tokens=True, max_length=config.max_len, truncation=True) for ja_token in
                ja_tokens]
    return bleu_score(candidate_corpus, reference_corpus)


if __name__ == '__main__':
    model = torch.load(config.model_path)
    test_loader = get_loader(test=True)

    bleu = eval(model)
    print(bleu)
