from gensim import models
from country_list import countries_for_language
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

w = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

countries_ = list(dict(countries_for_language('en')).values())
# print(countries)
countries = [country for country in countries_ if country in w]
countries_vec = [w[country] for country in countries if country in w]

model = TSNE(n_components=2)
countries_vec2D = model.fit_transform(countries_vec)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(countries_vec2D[:, 0], countries_vec2D[:, 1])
for i, text in enumerate(countries):
    print(i, text)
    ax.annotate(text, countries_vec2D[i])
plt.show()