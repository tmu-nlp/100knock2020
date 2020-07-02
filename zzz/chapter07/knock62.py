from gensim import models

w = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Find the top-N most similar words.
print(w.most_similar('United_States', topn=10))