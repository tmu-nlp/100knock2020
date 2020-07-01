'''
63. 加法構成性によるアナロジーPermalink
“Spain”の単語ベクトルから”Madrid”のベクトルを引き，”Athens”のベクトルを足したベクトルを計算し，
そのベクトルと類似度の高い10語とその類似度を出力せよ．
'''
import pickle
from pprint import pprint

with open("./models/google_model.pickle", mode="rb") as f:
    model = pickle.load(f)

pprint(model.most_similar(positive=["Spain", "Athens"], negative=["Madrid"]))

"""実行結果
[('Greece', 0.6898481249809265),
 ('Aristeidis_Grigoriadis', 0.5606848001480103), #ギリシャの水泳選手
 ('Ioannis_Drymonakos', 0.5552908778190613), #ギリシャの水泳選手
 ('Greeks', 0.545068621635437),
 ('Ioannis_Christou', 0.5400862693786621),
 ('Hrysopiyi_Devetzi', 0.5248444676399231),
 ('Heraklio', 0.5207759737968445),
 ('Athens_Greece', 0.516880989074707),
 ('Lithuania', 0.5166866183280945),
 ('Iraklion', 0.5146791934967041)]
"""
