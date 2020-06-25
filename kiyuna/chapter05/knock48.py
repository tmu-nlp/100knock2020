"""
48. 名詞から根へのパスの抽出
文中のすべての名詞を含む文節に対し，その文節から構文木の根に至るパスを抽出せよ．
ただし，構文木上のパスは以下の仕様を満たすものとする．

- 各文節は（表層形の）形態素列で表現する
- パスの開始文節から終了文節に至るまで，各文節の表現を” -> “で連結する

「ジョン・マッカーシーはAIに関する最初の会議で人工知能という用語を作り出した。」
という例文を考える． CaboChaを係り受け解析に用いた場合，次のような出力が得られると思われる．

    ジョンマッカーシーは -> 作り出した
    AIに関する -> 最初の -> 会議で -> 作り出した
    最初の -> 会議で -> 作り出した
    会議で -> 作り出した
    人工知能という -> 用語を -> 作り出した
    用語を -> 作り出した
"""
import os
import sys
from itertools import islice

from knock41 import cabocha_into_chunks
from knock45 import ChunkNormalized

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import message  # noqa: E402 isort:skip

if __name__ == "__main__":
    res = []
    # for chunks in cabocha_into_chunks():
    for chunks in islice(cabocha_into_chunks(), 33, 34):
        chunks = {k: ChunkNormalized(v) for k, v in chunks.items()}
        for c in chunks.values():
            if c.dst not in chunks:
                # 名詞を含まないのパス
                continue
            # [48-1] 各文節は（表層形の）形態素列で表現する
            tmp = [c.norm]
            # [48-2] パスの開始文節から終了文節に至るまで，各文節の表現を"->"で連結する
            dst = c.dst
            while dst in chunks:
                tmp.append(chunks[dst].norm)
                dst = chunks[dst].dst
            res.append(" -> ".join(tmp) + "\n")
    sys.stdout.writelines(res)
    message(f"write {len(res)} lines", type="success")
