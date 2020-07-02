"""
62. 類似度の高い単語10件
“United States”とコサイン類似度が高い10語と，その類似度を出力せよ．

[Ref]
- https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#word2vec-demo

[MEMO]
2015 年版の knock88 に対応
"""
import os
import sys
from pprint import pprint

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.pickle import load  # noqa: E402 isort:skip


if __name__ == "__main__":
    wv = load("chap07-embeddings")
    pprint(wv.most_similar(positive=["United_States"], topn=10))


"""result
[('Unites_States', 0.7877248525619507),
 ('Untied_States', 0.7541370391845703),
 ('United_Sates', 0.74007248878479),
 ('U.S.', 0.7310774326324463),
 ('theUnited_States', 0.6404393911361694),
 ('America', 0.6178410053253174),
 ('UnitedStates', 0.6167312264442444),
 ('Europe', 0.6132988929748535),
 ('countries', 0.6044804453849792),
 ('Canada', 0.6019070148468018)]
"""
