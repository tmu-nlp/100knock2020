import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

from knock60 import load_model
from knock67 import collect_target_vecs

if __name__ == '__main__':
    countries = []
    with open('./countries.txt') as fp:
        for line in fp:
            country = line.strip()
            countries.append(country)

    vecs, target_countries = collect_target_vecs(countries)

    plt.figure(figsize=(32.0, 24.0))
    link = linkage(vecs, method='ward')
    dendrogram(link, labels=target_countries,leaf_rotation=90,leaf_font_size=10)
    plt.show()
    plt.savefig('ward.png')

