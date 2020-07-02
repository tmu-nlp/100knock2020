from gensim.models import KeyedVectors
from sklearn import cluster
from sklearn import metrics

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

kmeans = cluster.KMeans(n_clusters = 5)
kmeans.fit(country_vectors)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print('Cluster id labels for inputted data')
print(labels)
print('\nCentroids data')
print(centroids)

'''
Cluster id labels for inputted data
[0 4 3 4 3 2 2 4 1 4 4 2 0 1 2 4 4 2 3 1 2 4 3 2 1 4 3 3 1 3 2 3 0 2 1 2 3
 3 2 4 2 4 4 4 3 2 2 1 2 0 2 3 3 4 3 1 4 4 3 3 4 4 3 4 2 2 3 2 2 2 4 4 1 1
 0 0 0 4 3 2 1 0 4 3 1 4 0 4 1 4 0 3 3 0 4 4 4 4 3 3 1 1 3 4 1 3 1 2 1 4 4
 1 4 0 3 1 3 1 1 4 1 2 3 3 4 0 0 1 2 2 2 1 4 4 0 4 3 2 2 2 1 4 3 0 3 4 3 3
 1 4 4 1 0 3 4 1 3 2 3 4 4 0 1 4 3 1 3 1 2 3 4 4 1 3 4 0 1 0 2 4 1 2 1 0 3
 3]

Centroids data
[[ 0.00947498  0.05927422 -0.04157221 ...  0.03868412  0.05498396
   0.11490595]
 [ 0.06118139 -0.08352322  0.0606397  ...  0.03253259  0.09740278
   0.18553162]
 [-0.10536463 -0.13955015  0.12132353 ... -0.06073267  0.18559759
   0.03063022]
 [ 0.03163945  0.09422601  0.03877952 ... -0.06754546 -0.04195087
   0.06887956]
 [ 0.08323999 -0.08870622  0.11799412 ...  0.05770238  0.16989255
   0.10221953]]
'''