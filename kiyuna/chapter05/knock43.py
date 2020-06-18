"""
43. 名詞を含む文節が動詞を含む文節に係るものを抽出
名詞を含む文節が，動詞を含む文節に係るとき，これらをタブ区切り形式で抽出せよ．
ただし，句読点などの記号は出力しないようにせよ．
"""
import os
import sys

from knock41 import Chunk, cabocha_into_chunks

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import message  # noqa: E402 isort:skip


class ChunkNormalized(Chunk):
    def __init__(self, chunk):
        self.morphs, self.dst, self.srcs = (*chunk,)
        self.norm = self.get_norm()

    def get_norm(self):
        clause = "".join(m.surface for m in self.morphs if m.pos != "記号")
        return clause

    def has_pos(self, pos):
        return any(m.pos == pos for m in self.morphs)


if __name__ == "__main__":
    res = []
    for chunks in cabocha_into_chunks():
        chunks = {k: ChunkNormalized(v) for k, v in chunks.items()}
        for c in chunks.values():
            if c.dst == -1:
                continue
            if c.dst not in chunks:
                continue
            if c.has_pos("名詞") and chunks[c.dst].has_pos("動詞"):
                res.append(f"{c.norm}\t{chunks[c.dst].norm}\n")
    sys.stdout.writelines(res)
    message(f"write {len(res)} lines", type="success")
