"""
65. アナロジータスクでの正解率
64の実行結果を用い，意味的アナロジー（semantic analogy）と
文法的アナロジー（syntactic analogy）の正解率を測定せよ．

[Ref]
- https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#word2vec-demo

[MEMO]
2015 年版の knock93 に対応
"""
import os
import sys
from enum import Enum

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer  # noqa: E402 isort:skip


class Analogy(Enum):
    Semantic = "semantic analogy"
    Syntactic = "syntactic analogy"

    def __str__(self):
        return self.value

    @classmethod
    def new(cls, line=""):
        if line.startswith(": gram"):
            return cls.Syntactic
        else:
            return cls.Semantic


if __name__ == "__main__":
    cnter = {Analogy.Semantic: [0, 0], Analogy.Syntactic: [0, 0]}
    analogy_type = None
    for line in open("./out64"):
        if line.startswith(":"):
            analogy_type = Analogy.new(line)
            continue
        w1, w2, w3, w4, w5, sim = line.split()
        cnter[analogy_type][w4 == w5] += 1
    with Renderer("knock65") as out:
        for member in Analogy:
            false, true = cnter[member]
            out.result(member, true / (false + true))


"""result
[*]  1. semantic analogy
0.7308602999210734
[*]  2. syntactic analogy
0.7400468384074942
"""
