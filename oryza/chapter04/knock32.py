from knock30 import conll_read
from knock31 import extract_postag

if __name__ == "__main__":
   print(extract_postag(conll_read(), 'lemma', 'VB'))