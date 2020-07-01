from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from knock60 import load_model
from knock67 import collect_target_vecs

if __name__ == '__main__':
    countries = []
    with open('./countries.txt') as fp:
        for line in fp:
            country = line.strip()
            countries.append(country)

    vecs, target_countries = collect_target_vecs(countries)

    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(vecs)

    vec_embedded = TSNE(n_components=2, random_state=0).fit_transform(vecs)
    fig, ax = plt.subplots(figsize=(16, 12))
    colors = ['red', 'blue', 'green', 'yellow', 'purple']
    for i, vec in enumerate(vec_embedded):
        plt.scatter(vec[0], vec[1], color=colors[kmeans.labels_[i]])
    for i, country in enumerate(target_countries):
        ax.annotate(country, (vec_embedded[i][0],vec_embedded[i][1]))
    plt.savefig('tsne.png')
    plt.show()

