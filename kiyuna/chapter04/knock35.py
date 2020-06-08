r"""knock35.py
35. 単語の出現頻度
文章中に出現する単語とその出現頻度を求め，出現頻度の高い順に並べよ．

[URL]
https://nlp100.github.io/ja/ch04.html#35-単語の出現頻度

[NOTE]
- 基本形にすると「だ」が上位にくるようになる

[Usage]
python knock35.py
"""
import os
import pprint
import sys
from typing import Counter, Dict, List

from tqdm import tqdm

from knock30 import mecab_into_sentences

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer  # noqa: E402 isort:skip

Morpheme = Dict[str, str]
Sentence = List[Morpheme]


def build_cnter(query: dict, *, verbose=False) -> Counter[str]:
    [(tgt_key, tgt_val)] = query.items()

    cnter = Counter()
    for sentence in tqdm(mecab_into_sentences()):
        cnter += Counter(d[tgt_key] for d in sentence)

    if verbose:
        with Renderer(f"「{tgt_val}」の出現頻度") as out:
            out.header("上位 10 個")
            pprint.pprint(cnter.most_common(10), stream=sys.stderr)
            out.result("種類", len(cnter))

    return cnter


if __name__ == "__main__":
    queries = [{"surface": "表層形"}, {"base": "基本形"}]

    for query in queries:
        build_cnter(query, verbose=True)
