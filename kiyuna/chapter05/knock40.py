"""
40. 係り受け解析結果の読み込み（形態素）
形態素を表すクラスMorphを実装せよ．このクラスは表層形（surface），基本形（base），
品詞（pos），品詞細分類1（pos1）をメンバ変数に持つこととする．
さらに，係り受け解析の結果（ai.ja.txt.parsed）を読み込み，
各文をMorphオブジェクトのリストとして表現し，冒頭の説明文の形態素列を表示せよ．

[Ref]
- https://taku910.github.io/cabocha/
    - mecab と同じ出力フォーマットっぽい
    - 表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音
"""
import pprint
import sys
from collections import namedtuple
from itertools import islice

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

Morph = namedtuple("Morph", ("surface", "base", "pos", "pos1"))


def cabocha_into_sentence(filename="ai.ja.txt.parsed"):
    sentence = []
    stack = 0
    with open(filename) as f:
        for line in map(lambda x: x.rstrip(), f):
            if line[0] == "*":
                continue
            if line == "EOS":
                assert stack == 0
                if sentence:
                    yield sentence
                sentence = []
                continue
            surface, details = line.split("\t")
            d = dict(zip(mecab_keys, details.split(",")))
            morph = Morph(
                surface=surface, base=d["原形"], pos=d["品詞"], pos1=d["品詞細分類1"],
            )
            sentence.append(morph)
            if d["品詞細分類1"] == "括弧開":
                stack += 1
                # print(stack, details, file=sys.stderr)
            if d["品詞細分類1"] == "括弧閉":
                stack -= 1
                # print(stack, details, file=sys.stderr)
                assert stack >= 0, details
            if surface == "。" and stack == 0:
                yield sentence
                sentence = []


def show_sentence(morphs, sep=""):
    print(sep.join(morph.surface for morph in morphs))


if __name__ == "__main__":
    print(sum(1 for line in cabocha_into_sentence()), "行", file=sys.stderr)
    for morphs in islice(cabocha_into_sentence(), 1, 2):
        pprint.pprint(morphs, stream=sys.stderr)
