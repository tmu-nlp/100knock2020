import sentencepiece as spm
import re
from fairseq.models.transformer import TransformerModel

sp = spm.SentencePieceProcessor()
sp.load("models/jsec.ja.model")

ja2en = TransformerModel.from_pretrained(
    'checkpoints/98subwords/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='data/bin/98_subwords/'
)

def raw2subword(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    subwords = sp.EncodeAsPieces(text)
    text = ' '.join(subwords)
    return text

def subword2raw(text):
    text = re.sub(' ', '', text)
    text = re.sub(r'▁', ' ', text)
    text = text[1:]
    return text.capitalize()
'''
def translate(text):
    text = raw2subword(text)
    text = ja2en.translate(text)
    return subword2raw(text)

'''

def translate(inputs):
    li = []
    inputs = inputs.split('\n')
    for text in inputs:
        text = raw2subword(text)
        text = ja2en.translate(text)
        li.append(subword2raw(text))
    return '\n'.join(li)