"""
45. 動詞の格パターンの抽出
今回用いている文章をコーパスと見なし，日本語の述語が取りうる格を調査したい．
動詞を述語，動詞に係っている文節の助詞を格と考え，述語と格をタブ区切り形式で出力せよ．
ただし，出力は以下の仕様を満たすようにせよ．

- 動詞を含む文節において，最左の動詞の基本形を述語とする
- 述語に係る助詞を格とする
- 述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる

「ジョン・マッカーシーはAIに関する最初の会議で人工知能という用語を作り出した。」
という例文を考える．この文は「作り出す」という１つの動詞を含み，
「作り出す」に係る文節は「ジョン・マッカーシーは」，「会議で」，「用語を」であると
解析された場合は，次のような出力になるはずである．

    作り出す	で は を

このプログラムの出力をファイルに保存し，以下の事項をUNIXコマンドを用いて確認せよ．

- コーパス中で頻出する述語と格パターンの組み合わせ
- 「行う」「なる」「与える」という動詞の格パターン（コーパス中で出現頻度の高い順に並べよ）
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

    def get_pos(self, pos):
        return (m for m in self.morphs if m.pos == pos)


if __name__ == "__main__":
    res = []
    for chunks in cabocha_into_chunks():
        chunks = {k: ChunkNormalized(v) for k, v in chunks.items()}
        for cv in filter(lambda c: c.has_pos("動詞"), chunks.values()):
            # 1. 動詞を含む文節において，最左の動詞の基本形を述語とする
            v_base = next(cv.get_pos("動詞")).base
            srcs = []
            for sc_idx in cv.srcs:
                # 2. 述語に係る助詞を格とする
                for m in chunks[sc_idx].get_pos("助詞"):
                    srcs.append(m.base)
            # 3. 述語に係る助詞（文節）が複数あるときは，
            #    すべての助詞をスペース区切りで辞書順に並べる
            if srcs:
                srcs.sort()
                particles = " ".join(srcs)
                res.append(f"{v_base}\t{particles}\n")
    sys.stdout.writelines(res)
    message(f"write {len(res)} lines", type="success")
