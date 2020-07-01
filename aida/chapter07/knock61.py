from gensim.models import KeyedVectors

from knock60 import load_model, obtain_vector

if __name__ == '__main__':
    model = load_model()
    src_word = 'United_States'
    tgt_word = 'U.S.'
    print(model.similarity(src_word, tgt_word))

"""
0.73107743
"""

