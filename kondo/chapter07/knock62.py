"""
62. 類似度の高い単語10件Permalink
“United States”とコサイン類似度が高い10語と，その類似度を出力せよ．
"""

import pickle
from gensim.models import word2vec

model_file = "model.sav"
with open(model_file, "rb") as file_model:
    model = pickle.load(file_model)

def find_topn(s, n):
    global model
    re = model.wv.most_similar(positive=s)
    re = re[0:n]
    return re

if __name__ == "__main__":
    re = find_topn("United_States", 10)
    for x in re:
        print(x)

"""
('Unites_States', 0.7877248525619507)
('Untied_States', 0.7541370391845703)
('United_Sates', 0.74007248878479)
('U.S.', 0.7310774326324463)
('theUnited_States', 0.6404393911361694)
('America', 0.6178410053253174)
('UnitedStates', 0.6167312264442444)
('Europe', 0.6132988929748535)
('countries', 0.6044804453849792)
('Canada', 0.6019070148468018)

出力までに時間かかった
"""