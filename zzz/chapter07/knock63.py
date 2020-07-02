from gensim import models

w = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
new_vec = w['Spain'] - w['Madrid'] + w['Athens']

# Find the top-N most similar words by vector.
print(w.similar_by_vector(new_vec))