from gensim import models
from country_list import countries_for_language
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


w = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

countries = list(dict(countries_for_language('en')).values())
# print(countries)
countries_vec = [w[country] for country in countries if country in w]

model = AgglomerativeClustering(n_clusters=5, linkage='ward')
model.fit(countries_vec)

fig = plt.figure()
ax = fig.add_subplot(111)

# plot the top three levels of the dendrogram
plot_dendrogram(model, labels=countries)

plt.show()
