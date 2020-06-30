from gensim.models import KeyedVectors
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
country_file = open('country-name.txt')

country_list = []
for country in country_file:
    country_list.append(country.strip())

country_vocab = []
country_vectors = []
for i in country_list:
    if i in model.vocab:
        country_vocab.append(i)
        country_vectors.append(model[i])

linked = linkage(country_vectors, 'ward')
plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            labels=country_vocab,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()