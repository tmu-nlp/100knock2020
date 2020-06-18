"""
47. 機能動詞構文のマイニング
動詞のヲ格にサ変接続名詞が入っている場合のみに着目したい．
46のプログラムを以下の仕様を満たすように改変せよ．

- 「サ変接続名詞+を（助詞）」で構成される文節が動詞に係る場合のみを対象とする
- 述語は「サ変接続名詞+を+動詞の基本形」とし，文節中に複数の動詞があるときは，
  最左の動詞を用いる
- 述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる
- 述語に係る文節が複数ある場合は，すべての項をスペース区切りで並べる（助詞の並び順と揃えよ）

例えば「また、自らの経験を元に学習を行う強化学習という手法もある。」という文から，
以下の出力が得られるはずである．

    学習を行う	に を	元に 経験を
"""
import os
import sys
from itertools import islice

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

    def get_pos(self, pos):
        return (m for m in self.morphs if m.pos == pos)

    def has_sahen_wo(self):
        for m1, m2 in zip(self.morphs, self.morphs[1:]):
            if [m1.pos1, m2.pos, m2.base] == ["サ変接続", "助詞", "を"]:
                return True
        return False


if __name__ == "__main__":
    res = []
    for chunks in islice(cabocha_into_chunks(), 12, 13):
        chunks = {k: ChunkNormalized(v) for k, v in chunks.items()}
        for cv in filter(lambda c: c.has_pos("動詞"), chunks.values()):
            # [45-1] 動詞を含む文節において，最左の動詞の基本形を述語とする
            v_base = next(cv.get_pos("動詞")).base
            srcs = []
            # [47-1] 「サ変接続名詞+を（助詞）」で構成される文節が動詞に係る場合のみを
            #        対象とする
            failed = True
            for sc_idx in reversed(cv.srcs):
                if failed and chunks[sc_idx].has_sahen_wo():
                    # [47-2] 述語は「サ変接続名詞+を+動詞の基本形」とし，
                    #        文節中に複数の動詞があるときは，最左の動詞を用いる
                    v_base = chunks[sc_idx].norm + v_base
                    failed = False
                else:
                    # [45-2] 述語に係る助詞を格とする
                    ms = tuple(chunks[sc_idx].get_pos("助詞"))
                    if ms:
                        # 文節内に助詞が複数ある場合は最も右のものを選ぶ
                        srcs.append((ms[0].base, chunks[sc_idx].norm))
            if failed:  # 「サ変接続名詞+を+動詞」がなかった
                continue
            if srcs:
                # [47-3] 述語に係る助詞（文節）が複数あるときは，
                #        すべての助詞をスペース区切りで辞書順に並べる
                # [47-4] 述語に係る文節が複数あるときは，
                #        助詞と同一の基準・順序でスペース区切りで並べる
                srcs.sort()
                pb, pc = zip(*srcs)
                p_base = " ".join(pb)
                # [46-1] 項は述語に係っている文節の単語列とする
                #        （末尾の助詞を取り除く必要はない）
                clause = " ".join(pc)
                res.append(f"{v_base}\t{p_base}\t{clause}\n")
    sys.stdout.writelines(res)
    message(f"write {len(res)} lines", type="success")

    # 1 つの chunk に複数の動詞を含む chunk を列挙
    # for i, chunks in enumerate(cabocha_into_chunks()):
    #     chunks = {k: ChunkNormalized(v) for k, v in chunks.items()}
    #     for chunk in chunks.values():
    #         if len(list(chunk.get_pos("動詞"))) > 1:
    #             print(i, chunk)
