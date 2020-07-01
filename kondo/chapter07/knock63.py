"""
63. 加法構成性によるアナロジーPermalink
“Spain”の単語ベクトルから”Madrid”のベクトルを引き，
”Athens”のベクトルを足したベクトルを計算し，
そのベクトルと類似度の高い10語とその類似度を出力せよ．
"""

import pickle
from gensim.models import word2vec

model_file = "model.sav"
with open(model_file, "rb") as file_model:
    model = pickle.load(file_model)

if __name__ == "__main__":
    results = model.wv.most_similar(positive=["Spain", "Athens"], negative=["Madrid"])
    results = results[:10]
    for result in results:
        print(result)

"""
('Greece', 0.6898481249809265)
('Aristeidis_Grigoriadis', 0.5606848001480103)
('Ioannis_Drymonakos', 0.5552908778190613)
('Greeks', 0.545068621635437)
('Ioannis_Christou', 0.5400862693786621)
('Hrysopiyi_Devetzi', 0.5248444676399231)
('Heraklio', 0.5207759737968445)
('Athens_Greece', 0.516880989074707)
('Lithuania', 0.5166866183280945)
('Iraklion', 0.5146791934967041)
"""