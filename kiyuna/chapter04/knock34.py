r"""knock34.py
34. 名詞の連接
名詞の連接（連続して出現する名詞）を最長一致で抽出せよ．

[URL]
https://nlp100.github.io/ja/ch04.html#34-名詞の連接

[Usage]
python knock34.py
"""
import os
import pprint
import sys
from typing import Dict, List

from knock30 import mecab_into_sentences

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer  # noqa: E402 isort:skip

Morpheme = Dict[str, str]
Sentence = List[Morpheme]


if __name__ == "__main__":
    tgt = "名詞+"

    sentinel: Morpheme = {"pos": "EOS"}

    res = []
    for sentence in mecab_into_sentences():
        sentence.append(sentinel)
        nouns = []
        for d in sentence:
            if d["pos"] == "名詞":
                nouns.append(d["surface"])
            else:
                if len(nouns) > 1:
                    res.append((len(nouns), "".join(nouns)))
                nouns = []

    with Renderer(tgt) as out:
        out.result("数", len(res))
        out.result("種類", len(set(res)))
        out.header("上から 5 個")
        pprint.pprint(res[:5], stream=sys.stderr)
        res.sort(reverse=True)
        out.header("大きい順に 10 個")
        pprint.pprint(res[:10], stream=sys.stderr)
