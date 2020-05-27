r"""knock06.py
06. 集合
“paraparaparadise”と”paragraph”に含まれる文字bi-gramの集合を，
それぞれ, XとYとして求め，XとYの和集合，積集合，差集合を求めよ．
さらに，’se’というbi-gramがXおよびYに含まれるかどうかを調べよ．

[URL]
https://nlp100.github.io/ja/ch01.html#06-集合

[Ref]
- set
    - https://docs.python.org/ja/3/library/stdtypes.html#set

[Usage]
python knock06.py
"""
import os
import sys

from knock05 import n_gram

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer, message  # noqa: E402 isort:skip


if __name__ == "__main__":

    w1 = "paraparaparadise"
    w2 = "paragraph"
    n = 2

    X = set(n_gram(w1, n))
    Y = set(n_gram(w2, n))
    tgt = n_gram("se", n).pop()

    print("X =", X)
    print("Y =", Y)

    print("X ∪ Y = {}".format(X | Y))  # X.union(Y)
    print("X ∩ Y = {}".format(X & Y))  # X.intersection(Y)
    print("X \\ Y = {}".format(X - Y))  # X.difference(Y)
    print("Y \\ X = {}".format(Y - X))  # Y.difference(X)

    print(f"X includes 'se': {tgt in X}")
    print(f"Y includes 'se': {tgt in Y}")

    with Renderer("MEMO") as out:
        out.result(r"X ∪ Y", X.union(Y))
        out.result(f"X ∩ Y", X.intersection(Y))
        out.result(rf"X \ Y", X.difference(n_gram(w2, n)))
        out.result(fr"Y \ X", Y.difference(n_gram(w1, n)))
