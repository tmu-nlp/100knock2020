from sklearn.cluster import KMeans

from knock60 import load_model

def collect_target_vecs(countries):
    not_found = 0
    vecs = []
    target_countries = []
    dic = {'United States of America': 'United_States', 'Russian Federation': 'Russia'}
    for country in countries:
        for k,v in dic.items():
            country = country.replace(k,v)
        country = country.replace(' ','_').replace('-','_').replace('_and_','_')
        try:
            vecs.append(model[country])
            target_countries.append(country)
        except:
            not_found += 1
    return vecs, target_countries

if __name__ == '__main__':
    countries = []
    with open('./countries.txt') as fp:
        for line in fp:
            country = line.strip()
            countries.append(country)

    vecs, target_countries = collect_target_vecs(countries)

    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(vecs)
    for label, country in sorted(zip(kmeans.labels_, target_countries)):
        print(f'{label}: {country}')

"""
0: Argentina
0: Bolivia
0: Brazil
0: Chile
0: Colombia
0: Costa_Rica
0: Cuba
0: Dominican_Republic
0: Ecuador
0: El_Salvador
0: Guatemala
0: Haiti
0: Honduras
0: Mexico
0: Nicaragua
0: Panama
0: Paraguay
0: Peru
0: Philippines
0: Uruguay
0: Venezuela
1: Afghanistan
1: Armenia
1: Azerbaijan
1: Bahrain
1: Bangladesh
1: Bhutan
1: Brunei_Darussalam
1: Cambodia
1: China
1: Egypt
1: India
1: Indonesia
1: Iran
1: Iraq
1: Israel
1: Japan
1: Jordan
1: Kazakhstan
1: Kuwait
1: Kyrgyzstan
1: Lebanon
1: Malaysia
1: Mongolia
1: Myanmar
1: Nepal
1: Oman
1: Pakistan
1: Qatar
1: Russia
1: Sao_Tome_Principe
1: Saudi_Arabia
1: Singapore
1: Syria
1: Tajikistan
1: Thailand
1: Turkey
1: Turkmenistan
1: United_Arab_Emirates
1: United_States
1: Uzbekistan
1: Viet_Nam
2: Albania
2: Andorra
2: Austria
2: Belarus
2: Belgium
2: Bosnia_Herzegovina
2: Bulgaria
2: Canada
2: Croatia
2: Cyprus
2: Czech_Republic
2: Denmark
2: Estonia
2: Finland
2: France
2: Georgia
2: Germany
2: Greece
2: Hungary
2: Iceland
2: Ireland
2: Italy
2: Latvia
2: Liechtenstein
2: Lithuania
2: Luxembourg
2: Malta
2: Monaco
2: Montenegro
2: Netherlands
2: Norway
2: Poland
2: Portugal
2: Romania
2: San_Marino
2: Serbia
2: Slovakia
2: Slovenia
2: Spain
2: Sweden
2: Switzerland
2: Ukraine
2: United_Kingdom
3: Algeria
3: Angola
3: Benin
3: Botswana
3: Burkina_Faso
3: Burundi
3: Cameroon
3: Chad
3: Comoros
3: Djibouti
3: Equatorial_Guinea
3: Eritrea
3: Ethiopia
3: Gabon
3: Gambia
3: Ghana
3: Guinea
3: Guinea_Bissau
3: Kenya
3: Lesotho
3: Liberia
3: Libya
3: Madagascar
3: Malawi
3: Mali
3: Mauritania
3: Morocco
3: Mozambique
3: Namibia
3: Niger
3: Nigeria
3: Rwanda
3: Senegal
3: Sierra_Leone
3: Somalia
3: South_Africa
3: Sudan
3: Togo
3: Tunisia
3: Uganda
3: Yemen
3: Zambia
3: Zimbabwe
4: Antigua_Barbuda
4: Australia
4: Bahamas
4: Barbados
4: Belize
4: Cabo_Verde
4: Dominica
4: Fiji
4: Grenada
4: Guyana
4: Jamaica
4: Kiribati
4: Maldives
4: Marshall_Islands
4: Mauritius
4: Nauru
4: New_Zealand
4: Palau
4: Saint_Lucia
4: Samoa
4: Seychelles
4: Solomon_Islands
4: Sri_Lanka
4: Suriname
4: Timor_Leste
4: Tonga
4: Trinidad_Tobago
4: Tuvalu
4: Vanuatu
"""

