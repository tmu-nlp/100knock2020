"""
41. 係り受け解析結果の読み込み（文節・係り受け）
40に加えて，文節を表すクラスChunkを実装せよ．
このクラスは形態素（Morphオブジェクト）のリスト（morphs），係り先文節インデックス番号（dst），
係り元文節インデックス番号のリスト（srcs）をメンバ変数に持つこととする．
さらに，入力テキストの係り受け解析結果を読み込み，
１文をChunkオブジェクトのリストとして表現し，冒頭の説明文の文節の文字列と係り先を表示せよ．

[Ref]
- https://taku910.github.io/cabocha/
    - CaboChaによる係り受け解析結果は，形態素解析結果の前に挿入される `*` から始まる行
    - * 1 2D 0/1 -0.764522          <- この行が文節の開始位置を意味している
        │ │   │   └─係り関係のスコア:
        │ │   │       係りやすさの度合を示します. 一般に大きな値ほど係りやすいことを表す
        │ │   └─ 主辞/機能語の位置
        │ └─ 係り先番号
        └─ 文節番号（0から始まる整数）
"""
import pprint
import sys
from collections import defaultdict, namedtuple
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

Morph = namedtuple("Morph", "surface, base, pos, pos1")


class Chunk:
    def __init__(self, morphs=[], dst=None, srcs=[]):
        self.morphs, self.dst, self.srcs = morphs[:], dst, srcs[:]

    def get_clause(self, sep=""):
        return sep.join(m.surface for m in self.morphs)

    def __repr__(self):
        clause = self.get_clause()
        return f"Chunk(srcs={self.srcs}, dst={self.dst}, morphs={clause})"

    def __getitem__(self, key):
        if key == 0:
            return self.morphs
        elif key == 1:
            return self.dst
        elif key == 2:
            return self.srcs
        else:
            raise IndexError


def cabocha_into_chunks(filename="ai.ja.txt.parsed"):
    chunks = defaultdict(Chunk)
    stack = 0
    with open(filename) as f:
        for line in map(lambda x: x.rstrip(), f):
            if line == "EOS":  # 文の終わり
                assert stack == 0
                if chunks:
                    yield {k: v for k, v in sorted(chunks.items()) if v.dst}
                    chunks.clear()
            elif line[0] == "*":  # 係り受けの情報
                _, idx, dst, *_ = line.split()
                idx = int(idx)
                dst = int(dst[:-1])
                chunks[idx].dst = dst
                chunks[dst].srcs.append(idx)
            else:  # 形態素の情報
                surface, details = line.split("\t")
                d = dict(zip(mecab_keys, details.split(",")))
                morph = Morph(
                    surface=surface,
                    base=d["原形"],
                    pos=d["品詞"],
                    pos1=d["品詞細分類1"],
                )
                chunks[idx].morphs.append(morph)
                if d["品詞細分類1"] == "括弧開":
                    stack += 1
                if d["品詞細分類1"] == "括弧閉":
                    stack -= 1
                    assert stack >= 0
                if surface == "。" and stack == 0:
                    yield {k: v for k, v in sorted(chunks.items()) if v.dst}
                    chunks.clear()


def show_chunks(chunks, sep=" "):
    print(sep.join(chunk.get_clause() for chunk in chunks.values()))


if __name__ == "__main__":
    print(sum(1 for line in cabocha_into_chunks()), "行", file=sys.stderr)
    for chunks in islice(cabocha_into_chunks(), 1, 2):
        pprint.pprint(chunks, stream=sys.stderr)
