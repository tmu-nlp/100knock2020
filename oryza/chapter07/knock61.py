from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
print(model.similarity('United_States','U.S.'))

'''
Similarity: 0.73107743
'''