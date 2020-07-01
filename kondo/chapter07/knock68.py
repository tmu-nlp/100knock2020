"""
68. Ward法によるクラスタリングPermalink
国名に関する単語ベクトルに対し，Ward法による階層型クラスタリングを実行せよ．
さらに，クラスタリング結果をデンドログラムとして可視化せよ．
"""

from scipy.cluster.hierarchy import linkage, dendrogram
from knock67 import make_dataframe, collect_countries
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    dataframe = make_dataframe(collect_countries())
    result = linkage(dataframe.iloc[:, 1:],
                        method="ward",
                        metric="euclidean")

    pd.set_option("display.max_rows", 116)
    #print(dataframe[0])
    #print(dataframe[0].values)
    a = dendrogram(result, labels=dataframe[0].values)
    plt.show()

"""
0        Afghanistan
1            Albania
2            Algeria
3             Angola
4            Armenia
5          Australia
6            Austria
7         Azerbaijan
8            Bahamas
9            Bahrain
10        Bangladesh
11           Belarus
12           Belgium
13            Belize
14            Bhutan
15          Botswana
16          Bulgaria
17           Burundi
18            Canada
19             Chile
20             China
21           Croatia
22              Cuba
23            Cyprus
24           Denmark
25          Dominica
26           Ecuador
27             Egypt
28           England
29           Eritrea
30           Estonia
31              Fiji
32           Finland
33            France
34             Gabon
35            Gambia
36           Georgia
37           Germany
38             Ghana
39            Greece
40         Greenland
41            Guinea
42            Guyana
43          Honduras
44           Hungary
45         Indonesia
46              Iran
47              Iraq
48           Ireland
49             Italy
50           Jamaica
51             Japan
52            Jordan
53        Kazakhstan
54             Kenya
55        Kyrgyzstan
56              Laos
57            Latvia
58           Lebanon
59           Liberia
60             Libya
61     Liechtenstein
62         Lithuania
63         Macedonia
64        Madagascar
65            Malawi
66              Mali
67             Malta
68        Mauritania
69           Moldova
70        Montenegro
71           Morocco
72        Mozambique
73           Namibia
74             Nepal
75         Nicaragua
76             Niger
77           Nigeria
78            Norway
79              Oman
80          Pakistan
81              Peru
82       Philippines
83            Poland
84          Portugal
85             Qatar
86           Romania
87            Russia
88            Rwanda
89             Samoa
90           Senegal
91            Serbia
92          Slovakia
93          Slovenia
94           Somalia
95             Spain
96             Sudan
97          Suriname
98            Sweden
99       Switzerland
100            Syria
101           Taiwan
102       Tajikistan
103         Thailand
104          Tunisia
105           Turkey
106     Turkmenistan
107           Tuvalu
108           Uganda
109          Ukraine
110          Uruguay
111       Uzbekistan
112        Venezuela
113          Vietnam
114           Zambia
115         Zimbabwe
"""