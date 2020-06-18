"""
46. 動詞の格フレーム情報の抽出
45のプログラムを改変し，述語と格パターンに続けて項（述語に係っている文節そのもの）を
タブ区切り形式で出力せよ．45の仕様に加えて，以下の仕様を満たすようにせよ．

- 項は述語に係っている文節の単語列とする（末尾の助詞を取り除く必要はない）
- 述語に係る文節が複数あるときは，助詞と同一の基準・順序でスペース区切りで並べる

「ジョン・マッカーシーはAIに関する最初の会議で人工知能という用語を作り出した。」
という例文を考える． この文は「作り出す」という１つの動詞を含み，
「作り出す」に係る文節は「ジョン・マッカーシーは」，「会議で」，「用語を」であると
解析された場合は，次のような出力になるはずである．

    作り出す	で は を	会議で ジョンマッカーシーは 用語を
"""
import os
import sys
from itertools import islice

from knock41 import Chunk, cabocha_into_chunks

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import message  # noqa: E402 isort:skip


class ChunkNormalized(Chunk):  # knock45 と同様
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


if __name__ == "__main__":
    res = []
    # for chunks in cabocha_into_chunks():
    for chunks in islice(cabocha_into_chunks(), 33, 34):  # list[33:34] と同様
        chunks = {k: ChunkNormalized(v) for k, v in chunks.items()}
        for cv in filter(lambda c: c.has_pos("動詞"), chunks.values()):
            # [45-1] 動詞を含む文節において，最左の動詞の基本形を述語とする
            v_base = next(cv.get_pos("動詞")).base
            srcs = []
            for sc_idx in cv.srcs:
                # [45-2] 述語に係る助詞を格とする
                ms = tuple(chunks[sc_idx].get_pos("助詞"))
                if ms:
                    # 文節内に助詞が複数ある場合は最も右のものを選ぶ
                    # e.g. 「次のように」
                    #   Morph(surface='の', base='の', pos='助詞', pos1='連体化')
                    #   Morph(surface='に', base='に', pos='助詞', pos1='副詞化')
                    # if len(ms) != 1:
                    #     print(chunks[sc_idx].norm, ms)
                    srcs.append((ms[-1].base, chunks[sc_idx].norm))
            if srcs:
                # [45-3] 述語に係る助詞（文節）が複数あるときは，
                #        すべての助詞をスペース区切りで辞書順に並べる
                # [46-2] 述語に係る文節が複数あるときは，
                #        助詞と同一の基準・順序でスペース区切りで並べる
                srcs.sort()
                pbs, pcs = zip(*srcs)
                p_bases = " ".join(pbs)
                # [46-1] 項は述語に係っている文節の単語列とする
                #        （末尾の助詞を取り除く必要はない）
                clauses = " ".join(pcs)
                res.append(f"{v_base}\t{p_bases}\t{clauses}\n")
    sys.stdout.writelines(res)
    message(f"write {len(res)} lines", type="success")
