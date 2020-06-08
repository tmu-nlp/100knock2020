r"""knock33.py
33. 「AのB」
2つの名詞が「の」で連結されている名詞句を抽出せよ．

[URL]
https://nlp100.github.io/ja/ch04.html#33-aのb

[Usage]
python knock33.py
"""
import os
import sys
from typing import Dict, List

from knock30 import mecab_into_sentences

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer  # noqa: E402 isort:skip

Morpheme = Dict[str, str]
Sentence = List[Morpheme]


if __name__ == "__main__":
    tgt = "AのB"

    res = []
    for sentence in mecab_into_sentences():
        for a, no, b in zip(sentence, sentence[1:], sentence[2:]):
            if (a["pos"], no["surface"], b["pos"]) == ("名詞", "の", "名詞"):
                res.append("".join(map(lambda x: x["surface"], (a, no, b))))

    with Renderer(tgt) as out:
        out.result("数", len(res))
        out.result("種類", len(set(res)))
        out.result("上から 10 個", res[:10])
