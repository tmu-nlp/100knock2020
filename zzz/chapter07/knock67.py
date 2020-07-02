from gensim import models
from country_list import countries_for_language
from sklearn.cluster import KMeans

w = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
countries = list(dict(countries_for_language('en')).values())
# print(countries)
countries_vec = [w[country] for country in countries if country in w]

model = KMeans(5)

model.fit(countries_vec)

print(countries[:5], model.predict(countries_vec[:5]))