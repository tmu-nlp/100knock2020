# 67. k-meansクラスタリング
# 国名に関する単語ベクトルを抽出し，k-meansクラスタリングをクラスタ数k=5として実行せよ．

from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # データを読み込む
    model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

    # 以下のWebページにある国名を使う
    # http://www.fao.org/countryprofiles/iso3list/en/
    countries = pd.read_table("countries.tsv")
    countries = countries["Short name"].values

    # 国名のベクトルを取り出す
    country_vec = []
    country_name = []
    for country in countries:
        if country in model.vocab:
            country_vec.append(model[country])
            country_name.append(country)

    # k-meansクラスタリング
    km = KMeans(n_clusters=5, random_state=0)
    y_km = km.fit_predict(country_vec)

    # 結果を表示する
    for country, cluster in zip(country_name, y_km):
        print(f"{country}: {cluster}")


# 結果
# Afghanistan: 2
# Albania: 0
# Algeria: 3
# Andorra: 1
# Angola: 3
# Argentina: 1
# Armenia: 0
# Australia: 1
# Austria: 1
# Azerbaijan: 0
# Bahamas: 4
# Bahrain: 2
# Bangladesh: 2
# Barbados: 4
# Belarus: 0
# Belgium: 1
# Belize: 4
# Benin: 3
# Bhutan: 2
# Botswana: 3
# Brazil: 1
# Bulgaria: 0
# Burundi: 3
# Cambodia: 2
# Cameroon: 3
# Canada: 1
# Chad: 3
# Chile: 1
# China: 2
# Colombia: 1
# Comoros: 3
# Congo: 3
# Croatia: 0
# Cuba: 1
# Cyprus: 0
# Czechia: 0
# Denmark: 1
# Djibouti: 3
# Dominica: 4
# Ecuador: 1
# Egypt: 2
# Eritrea: 3
# Estonia: 0
# Ethiopia: 3
# Fiji: 4
# Finland: 1
# France: 1
# Gabon: 3
# Gambia: 3
# Georgia: 0
# Germany: 1
# Ghana: 3
# Greece: 0
# Grenada: 4
# Guatemala: 1
# Guinea: 3
# Guyana: 4
# Haiti: 3
# Honduras: 1
# Hungary: 0
# Iceland: 0
# India: 2
# Indonesia: 2
# Iraq: 2
# Ireland: 1
# Israel: 2
# Italy: 1
# Jamaica: 4
# Japan: 1
# Jordan: 2
# Kazakhstan: 0
# Kenya: 3
# Kiribati: 4
# Kuwait: 2
# Kyrgyzstan: 2
# Latvia: 0
# Lebanon: 2
# Lesotho: 3
# Liberia: 3
# Libya: 2
# Lithuania: 0
# Luxembourg: 1
# Madagascar: 3
# Malawi: 3
# Malaysia: 2
# Maldives: 4
# Mali: 3
# Malta: 0
# Mauritania: 3
# Mauritius: 4
# Mexico: 1
# Monaco: 1
# Mongolia: 2
# Montenegro: 0
# Morocco: 1
# Mozambique: 3
# Myanmar: 2
# Namibia: 3
# Nauru: 4
# Nepal: 2
# Netherlands: 1
# Nicaragua: 1
# Niger: 3
# Nigeria: 3
# Niue: 4
# Norway: 1
# Oman: 2
# Pakistan: 2
# Palau: 4
# Panama: 1
# Paraguay: 1
# Peru: 1
# Philippines: 1
# Poland: 0
# Portugal: 1
# Qatar: 2
# Romania: 0
# Rwanda: 3
# Samoa: 4
# Senegal: 3
# Serbia: 0
# Seychelles: 4
# Singapore: 2
# Slovakia: 0
# Slovenia: 0
# Somalia: 3
# Spain: 1
# Sudan: 3
# Suriname: 4
# Sweden: 1
# Switzerland: 1
# Tajikistan: 2
# Thailand: 2
# Togo: 3
# Tokelau: 4
# Tonga: 4
# Tunisia: 3
# Turkey: 0
# Turkmenistan: 2
# Tuvalu: 4
# Uganda: 3
# Ukraine: 0
# Uruguay: 1
# Uzbekistan: 2
# Vanuatu: 4
# Yemen: 2
# Zambia: 3
# Zimbabwe: 3
