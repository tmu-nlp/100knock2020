'''
67. k-meansクラスタリングPermalink
国名に関する単語ベクトルを抽出し，k-meansクラスタリングをクラスタ数k=5として実行せよ．
'''
import pickle
from pprint import pprint
from sklearn.cluster import KMeans

with open("./models/google_model.pickle", mode="rb") as f:
    model = pickle.load(f)

# 国名の集合
countries = set()
with open("./data/questions-words.txt") as f:
    for line in f:
        if line[0] == ":":
            cols = line.split()
            # capitalから始まるもの
            if cols[1].startswith("capital"):
                flag = True
            else:
                flag = False
        else:
            if flag:
                cols = line.split()
                # １番目と３番目は国名
                countries.add(cols[1])
                countries.add(cols[3])
            else:
                break
#countries = list(countries)

# ベクトルを得る
countries_vectors = []
for country in countries:
    vector = model[country]
    countries_vectors.append(vector)

# 辞書を初期化
dic = {}
for i in range(5):
    dic[f"class {i}"] = []

# インスタンス作成
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(countries_vectors)
# 辞書に国を突っ込んでいく
for country, label in zip(countries, kmeans.labels_):
    dic[f"class {label}"].append(country)

for i in range(5):
    print(f"class {i}")
    print(", ".join(dic[f"class {i}"]))

with open('./data/country_names.pickle', mode='wb') as f1\
    , open('./data/country_vectors.pickle', mode='wb') as f2\
    , open('./data/cv_dic.pickle', mode='wb') as f3:
    pickle.dump(countries, f1)
    pickle.dump(countries_vectors, f2)
    pickle.dump(dic, f3)

'''
class 0
Niger, Mali, Malawi, Mozambique, Kenya, Tunisia, Liberia, Namibia, Zambia, Somalia, Mauritania, Sudan, Senegal, Zimbabwe, Ghana, Rwanda, Burundi, Gabon, Angola, Uganda, Guinea, Eritrea, Algeria, Botswana, Madagascar, Gambia, Nigeria
class 1
Lebanon, Taiwan, Egypt, Bhutan, Japan, Afghanistan, Syria, Qatar, Greenland, Morocco, Thailand, Libya, Vietnam, Philippines, Indonesia, Tuvalu, Nepal, Fiji, Jordan, Pakistan, Bahrain, Iran, Iraq, Bangladesh, China, Laos, Oman
class 2
Turkmenistan, Turkey, Bulgaria, Cyprus, Georgia, Belarus, Albania, Russia, Uzbekistan, Montenegro, Armenia, Malta, Kazakhstan, Azerbaijan, Serbia, Ukraine, Moldova, Kyrgyzstan, Greece, Macedonia, Romania, Tajikistan
class 3
Italy, Lithuania, Poland, Sweden, Austria, Denmark, Finland, Latvia, Estonia, France, Belgium, Liechtenstein, Switzerland, Norway, Croatia, Slovakia, Hungary, Slovenia, Germany
class 4
Dominica, Nicaragua, Peru, Bahamas, Uruguay, Guyana, Venezuela, Portugal, Belize, Spain, Ireland, Samoa, England, Ecuador, Jamaica, Canada, Chile, Honduras, Australia, Cuba, Suriname
'''