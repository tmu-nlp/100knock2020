"""
42. 係り元と係り先の文節の表示
係り元の文節と係り先の文節のテキストをタブ区切り形式ですべて抽出せよ．
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


if __name__ == "__main__":
    res = []
    for chunks in cabocha_into_chunks():
        chunks = {k: ChunkNormalized(v) for k, v in chunks.items()}
        for c in chunks.values():
            if c.dst == -1:  # （注）python だとインデックスの -1 は
                continue  # エラーにならず，リストの最後が選ばれてしまう
            if c.norm and c.dst in chunks and chunks[c.dst].norm:
                print(c.norm, chunks[c.dst].norm)
                res.append(f"{c.norm}\t{chunks[c.dst].norm}\n")
    sys.stdout.writelines(res)
    message(f"write {len(res)} lines", type="success")

    """記号の一覧を表示する
    from knock40 import cabocha_into_sentence

    symbols = {
        m.base for m in sum(cabocha_into_sentence(), []) if m.pos == "記号"
    }
    print(symbols, file=sys.stderr)

    # {'（', '〉', '〔', '〈', '、', '・', '」', '＝', '々', '「', '）', '』', '*', '，', '？', '〕', '！', '『', '。'}
    """
