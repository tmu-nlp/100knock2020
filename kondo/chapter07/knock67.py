"""
67. k-meansクラスタリングPermalink
国名に関する単語ベクトルを抽出し，k-meansクラスタリングをクラスタ数k=5として実行せよ．
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans

analogy_file = "analogy_data"

model_file = "model.sav"
with open(model_file, "rb") as file_model:
    model = pickle.load(file_model)

def collect_countries():
    countries = set()
    with open(analogy_file, encoding="utf-8") as file_analogy:
        for line in file_analogy:
            words = line.split()
            if words[0] == ":" and words[1] == "currency":
                break
            if words[0] == ":":
                continue
            countries.add(words[1])
    countries = list(countries)
    countries.sort()
    df = pd.DataFrame(countries)
    return df

def make_dataframe(words_df):
    vec_list = []
    for country in words_df[0]:
        vec_list.append(model[str(country)])
    vec_list = np.array(vec_list)
    #print(vec_list.shape[1])
    vec = pd.DataFrame(vec_list, columns=range(1, vec_list.shape[1]+1))
    dataframe = pd.concat([words_df, vec], axis=1)
    return dataframe

if __name__ == "__main__":
    countries_df = collect_countries()
    dataframe = make_dataframe(countries_df)
    kmeans_model = KMeans(n_clusters=5, random_state=10).fit(dataframe.iloc[:, 1:])
    labels = kmeans_model.labels_

    print(labels)

    for i in range(5):
        print(f"cluster{i}")
        index = np.where(labels == i)[0]
        #print(index)
        print(", ".join([dataframe[0][j] for j in index]))

"""
cluster0
Albania, Austria, Belgium, Bulgaria, Croatia, Cyprus, Denmark, England, Estonia, Finland, France, Germany, Greece, Hungary, Ireland, Italy, Latvia, Liechtenstein, Lithuania, Macedonia, Malta, Montenegro, Norway, Poland, Portugal, Romania, Serbia, Slovakia, Slovenia, Spain, Sweden, Switzerland, Uruguay
cluster1
Armenia, Azerbaijan, Belarus, Kazakhstan, Kyrgyzstan, Moldova, Russia, Tajikistan, Turkmenistan, Ukraine, Uzbekistan
cluster2
Australia, Bahamas, Bangladesh, Belize, Bhutan, Canada, Chile, China, Cuba, Ecuador, Fiji, Georgia, Greenland, Honduras, Indonesia, Japan, Laos, Nepal, Nicaragua, Peru, Philippines, Samoa, Taiwan, Thailand, Tuvalu, Venezuela, Vietnam
cluster3
Afghanistan, Algeria, Bahrain, Egypt, Eritrea, Iran, Iraq, Jordan, Lebanon, Libya, Mauritania, Morocco, Niger, Oman, Pakistan, Qatar, Somalia, Sudan, Syria, Tunisia, Turkey
cluster4
Angola, Botswana, Burundi, Dominica, Gabon, Gambia, Ghana, Guinea, Guyana, Jamaica, Kenya, Liberia, Madagascar, Malawi, Mali, Mozambique, Namibia, Nigeria, Rwanda, Senegal, Suriname, Uganda, Zambia, Zimbabwe
"""