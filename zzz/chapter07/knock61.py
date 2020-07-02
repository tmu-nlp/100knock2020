from gensim import models

w = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Compute cosine similarity between two words.
print(w.similarity('United_States', 'U.S.'))