"""
61. 単語の類似度
“United States”と”U.S.”のコサイン類似度を計算せよ．

[Ref]
- https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#word2vec-demo

[MEMO]
2015 年版の knock87 に対応
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.pickle import load  # noqa: E402 isort:skip


if __name__ == "__main__":
    wv = load("chap07-embeddings")
    print(wv.similarity("United_States", "U.S."))


"""result
0.73107743
"""
