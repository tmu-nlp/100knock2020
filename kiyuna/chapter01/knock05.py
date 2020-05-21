r"""knock05.py
05. n-gram
与えられたシーケンス（文字列やリストなど）からn-gramを作る関数を作成せよ．
この関数を用い，”I am an NLPer”という文から単語bi-gram，文字bi-gramを得よ．

[URL]
https://nlp100.github.io/ja/ch01.html#05-n-gram

[Usage]
python knock05.py
"""
import os
import sys
from typing import Any, List, Sequence

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import kiyuna.common as my  # isort:skip
from kiyuna.utils.message import Renderer, message  # isort:skip


def n_gram(seq: Sequence[Any], n: int) -> List[Sequence[Any]]:
    return [seq[i : i + n] for i in range(len(seq) - n + 1)]


if __name__ == "__main__":

    sent = "I am an NLPer"

    with Renderer("knock05") as out:

        words = sent.split()
        chars = sent
        n = 2

        out.result("word bi-gram", n_gram(words, n))
        out.result("letter bi-grams", n_gram(chars, n))

        out.result("word bi-gram (w/ zip)", my.n_gram(words, n))
        out.result("letter bi-grams (w/ zip)", my.n_gram(chars, n))
