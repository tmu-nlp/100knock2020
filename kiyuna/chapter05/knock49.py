"""
49. 名詞間の係り受けパスの抽出
文中のすべての名詞句のペアを結ぶ最短係り受けパスを抽出せよ．
ただし，名詞句ペアの文節番号がiとj（i<j）のとき，係り受けパスは以下の仕様を満たすものとする．

- 問題48と同様に，パスは開始文節から終了文節に至るまでの各文節の表現（表層形の形態素列）を
  ” -> “で連結して表現する
- 文節iとjに含まれる名詞句はそれぞれ，XとYに置換する

また，係り受けパスの形状は，以下の2通りが考えられる．

- 文節iから構文木の根に至る経路上に文節jが存在する場合: 文節iから文節jのパスを表示
- 上記以外で，文節iと文節jから構文木の根に至る経路上で共通の文節kで交わる場合:
    文節iから文節kに至る直前のパスと文節jから文節kに至る直前までのパス，
    文節kの内容を” | “で連結して表示

「ジョン・マッカーシーはAIに関する最初の会議で人工知能という用語を作り出した。」
という例文を考える． CaboChaを係り受け解析に用いた場合，次のような出力が得られると思われる．

    Xは | Yに関する -> 最初の -> 会議で | 作り出した
    Xは | Yの -> 会議で | 作り出した
    Xは | Yで | 作り出した
    Xは | Yという -> 用語を | 作り出した
    Xは | Yを | 作り出した
    Xに関する -> Yの
    Xに関する -> 最初の -> Yで
    Xに関する -> 最初の -> 会議で | Yという -> 用語を | 作り出した
    Xに関する -> 最初の -> 会議で | Yを | 作り出した
    Xの -> Yで
    Xの -> 会議で | Yという -> 用語を | 作り出した
    Xの -> 会議で | Yを | 作り出した
    Xで | Yという -> 用語を | 作り出した
    Xで | Yを | 作り出した
    Xという -> Yを
"""
import os
import sys
from itertools import combinations, islice

from knock41 import Chunk, cabocha_into_chunks

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import message  # noqa: E402 isort:skip

# d = dict()


class ChunkNormalized(Chunk):
    def __init__(self, chunk):
        self.morphs, self.dst, self.srcs = (*chunk,)
        self.norm = self.get_norm()

    def has_pos(self, pos):
        return any(m.pos == pos for m in self.morphs)

    def get_pos(self, pos):
        return (m for m in self.morphs if m.pos == pos)

    def list_pos(self, pos):
        return [i for i, m in enumerate(self.morphs) if m.pos == pos]

    def get_norm(self):
        # [48-1] 各文節は（表層形の）形態素列で表現する
        clause = "".join(m.surface for m in self.morphs if m.pos != "記号")
        return clause

    def replace_noun(self, repl):
        """
        [49-2] 文節iとjに含まれる名詞句はそれぞれ，XとYに置換する
            e.g. 3 行目
                ('名詞', '名詞', '名詞', '助詞') 佐藤理史は
                ('名詞', '助詞', '名詞', '助詞') 次のように
        """
        idxs = self.list_pos("名詞")
        l, r = idxs[0], idxs[-1] + 1  # half-open interval
        tmp = []
        tmp += [m.surface for m in self.morphs[:l] if m.pos != "記号"]
        tmp.append(repl)  # self.morphs[l, r) はまとめて repl に置換
        tmp += [m.surface for m in self.morphs[r:] if m.pos != "記号"]
        clause = "".join(tmp)
        # d[tuple(m.pos for m in self.morphs)] = self.norm
        return clause


def replace_and_concat(idxs, chunks):
    head, *body, tail = map(lambda x: chunks[x], idxs)
    res = []
    res += [head.replace_noun("X")]
    res += [chunk.norm for chunk in body]
    res += [tail.replace_noun("Y")]
    # [49-1] パスは開始文節から終了文節に至るまでの各文節の表現（表層形の形態素列）を
    #        ” -> “で連結して表現する
    return " -> ".join(res)


def replace_and_combine(part1, part2, tail, chunks):
    res = [None] * 3
    for i, part in (0, part1), (1, part2):
        head, *body = map(lambda idx: chunks[idx], part)
        tmp = [head.replace_noun("XY"[i])] + [chunk.norm for chunk in body]
        res[i] = " -> ".join(tmp)
    res[2] = chunks[tail].norm
    # [49-1] パスは開始文節から終了文節に至るまでの各文節の表現（表層形の形態素列）を
    #        ” -> “で連結して表現する
    return " | ".join(res)


if __name__ == "__main__":
    res = []
    # for chunks in cabocha_into_chunks():
    for chunks in islice(cabocha_into_chunks(), 33, 34):
        chunks = {k: ChunkNormalized(v) for k, v in chunks.items()}
        paths = []
        # 名詞列のパス（idx の列）を作る
        for idx, chunk in chunks.items():
            if not chunk.has_pos("名詞"):
                continue
            if chunk.dst not in chunks:  # 名詞を含まないのパス
                continue
            tmp = [idx]
            dst = chunk.dst
            while dst in chunks:
                tmp.append(dst)
                dst = chunks[dst].dst
            paths.append(tmp)
        # i < j なる 2 つのパスを列挙
        for p1, p2 in combinations(paths, 2):
            p2_head = p2[0]  # 文節 j
            if p2_head in p1:
                """
                [49-3] 文節iから構文木の根に至る経路上に文節jが存在する場合:
                    文節iから文節jのパスを表示
                        [1, 2, 3, 4, 5] [3, 4, 5] -> [1, 2, 3]
                """
                p1_tail = p1.index(p2_head)
                idxs = p1[: p1_tail + 1]
                ans = replace_and_concat(idxs, chunks)
            else:
                """
                [49-4] 上記以外で，
                    文節iと文節jから構文木の根に至る経路上で共通の文節kで交わる場合:
                    文節iから文節kに至る直前のパスと文節jから文節kに至る直前までのパス，
                    文節kの内容を"|"で連結して表示
                        [0, 5] [1, 2, 3, 4, 5] -> [0] [1, 2, 3, 4] [5]
                        [0, 4, 5] [3, 4, 5] -> [0] [3] [4]
                """
                for i, w1 in enumerate(p1[1:], start=1):
                    if w1 in p2:
                        part1, tail = p1[:i], p1[i]
                        part2 = p2[: p2.index(w1)]
                        break
                ans = replace_and_combine(part1, part2, tail, chunks)
            res.append(ans + "\n")
    sys.stdout.writelines(res)
    message(f"write {len(res)} lines", type="success")

    # 1 つの chunk に複数の名詞を含む chunk を列挙
    # for k, v in d.items():
    #     if k.count("名詞") > 1:
    #         print(k, v, file=sys.stderr)
