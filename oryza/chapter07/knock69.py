from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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

tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=0)
new_values = tsne_model.fit_transform(country_vectors)

x = []
y = []
for value in new_values:
    x.append(value[0])
    y.append(value[1])
    
plt.figure(figsize=(10, 10)) 
for i in range(len(x)):
    plt.scatter(x[i],y[i])
    plt.annotate(country_vocab[i],
                    xy=(x[i], y[i]),
                    xytext=(5, 2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom')
plt.show()