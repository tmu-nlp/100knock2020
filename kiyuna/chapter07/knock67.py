"""
67. k-meansクラスタリング
国名に関する単語ベクトルを抽出し，k-meansクラスタリングをクラスタ数k=5として実行せよ．

[Ref]
- sklearn.cluster.KMeans
    - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

[MEMO]
2015 年版の knock96-97 に対応
"""
import os
import sys
import numpy as np
from sklearn.cluster import KMeans

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer  # noqa: E402 isort:skip
from kiyuna.utils.pickle import dump, load  # noqa: E402 isort:skip


def list_country_names(path="./questions-words.txt"):
    country_names = set()
    adding = False
    with open(path) as f:
        for line in f:
            if line.startswith(":"):
                adding = line in (": capital-common-countries\n", ": capital-world\n")
                continue
            if adding:
                capital1, country1, capital2, country2 = line.split()
                country_names |= {country1, country2}
    return list(country_names)


def country_embeddings():
    wv = load("chap07-embeddings")
    country_names = np.array(list_country_names(), dtype=object)
    embeddings = [wv[country_name] for country_name in country_names]
    dump([embeddings, country_names], "chap07-embeddings-country")
    return embeddings, country_names


if __name__ == "__main__":
    embeddings, country_names = country_embeddings()
    kmeans = KMeans(n_clusters=5).fit(embeddings)
    dump(kmeans, "chap07-kmeans")
    with Renderer("knock67", start=0) as out:
        for i in range(5):
            out.result(f"Class {i}", country_names[kmeans.labels_ == i])


"""result
[*]  0. Class 0
['Bhutan' 'Bahrain' 'Japan' 'Morocco' 'Indonesia' 'Pakistan' 'Thailand'
 'Tunisia' 'Oman' 'Egypt' 'Turkey' 'Qatar' 'Iraq' 'Laos' 'Libya' 'Lebanon'
 'Jordan' 'Afghanistan' 'Bangladesh' 'Syria' 'Nepal' 'China' 'Vietnam'
 'Iran']
[*]  1. Class 1
['Samoa' 'Chile' 'Dominica' 'Australia' 'Ecuador' 'Fiji' 'Bahamas'
 'Canada' 'Jamaica' 'Nicaragua' 'Cuba' 'Peru' 'Venezuela' 'Uruguay'
 'Guyana' 'Honduras' 'Belize' 'Greenland' 'Philippines' 'Taiwan' 'Tuvalu'
 'Suriname']
[*]  2. Class 2
['Ghana' 'Malawi' 'Gabon' 'Gambia' 'Namibia' 'Guinea' 'Uganda' 'Somalia'
 'Mauritania' 'Angola' 'Sudan' 'Niger' 'Madagascar' 'Zimbabwe' 'Botswana'
 'Burundi' 'Mali' 'Senegal' 'Kenya' 'Liberia' 'Eritrea' 'Mozambique'
 'Zambia' 'Nigeria' 'Rwanda' 'Algeria']
[*]  3. Class 3
['Albania' 'Latvia' 'Norway' 'Portugal' 'Macedonia' 'Liechtenstein'
 'Slovakia' 'Montenegro' 'France' 'Cyprus' 'Spain' 'Lithuania' 'Bulgaria'
 'Austria' 'Germany' 'Finland' 'Croatia' 'Ireland' 'Estonia' 'Romania'
 'Poland' 'Italy' 'Slovenia' 'Sweden' 'Greece' 'Denmark' 'Serbia' 'Malta'
 'Switzerland' 'Hungary' 'Belgium' 'England']
[*]  4. Class 4
['Russia' 'Turkmenistan' 'Ukraine' 'Kazakhstan' 'Azerbaijan' 'Belarus'
 'Moldova' 'Georgia' 'Armenia' 'Tajikistan' 'Kyrgyzstan' 'Uzbekistan']
"""
