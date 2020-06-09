r"""knock30.py
30. 形態素解析結果の読み込み
形態素解析結果（neko.txt.mecab）を読み込むプログラムを実装せよ．
ただし，各形態素は表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）を
キーとするマッピング型に格納し，1文を形態素（マッピング型）のリストとして表現せよ．

[URL]
https://nlp100.github.io/ja/ch04.html#30-形態素解析結果の読み込み

[Ref]
- MeCab の出力フォーマット
    - https://taku910.github.io/mecab/
        - 表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音

[Usage]
python knock30.py
"""
import os
import pprint
import sys
from itertools import islice
from typing import Dict, Iterator, List

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer  # noqa: E402 isort:skip

Morpheme = Dict[str, str]
Sentence = List[Morpheme]


def mecab_into_sentences(path: str = "./neko.txt.mecab") -> Iterator[Sentence]:
    sentence: Sentence = []
    with open(path) as f:
        for line in map(lambda x: x.rstrip(), f):
            if not line:
                continue
            if line == "EOS":
                yield sentence
                sentence = []
                continue
            surface, details = line.split("\t")
            mecab_keys = [
                "品詞",
                "品詞細分類1",
                "品詞細分類2",
                "品詞細分類3",
                "活用型",
                "活用形",
                "原形",
                "読み",
                "発音",
            ]
            d = dict(zip(mecab_keys, details.split(",")))
            morpheme: Morpheme = {
                "surface": surface,
                "base": d["原形"],
                "pos": d["品詞"],
                "pos1": d["品詞細分類1"],
            }
            sentence.append(morpheme)


def test_extract(query: dict, *, verbose=False) -> list:
    [(src_key, src_val)] = query["src"].items()
    [(dst_key, dst_val)] = query["dst"].items()

    res = []
    for sentence in mecab_into_sentences():
        res.extend([d[dst_key] for d in sentence if d[src_key] == src_val])

    if verbose:
        with Renderer(f"「{src_val}」の「{dst_val}」") as out:
            out.result("数", len(res))
            out.result("種類", len(set(res)))
            out.result("上から 10 個", res[:10])

    return res


if __name__ == "__main__":
    filename = "neko.txt.mecab"
    for sentence in islice(mecab_into_sentences(filename), 2, 4):
        pprint.pprint(sentence)
