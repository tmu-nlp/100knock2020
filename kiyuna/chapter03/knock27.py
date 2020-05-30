r"""knock27.py
27. 内部リンクの除去
26の処理に加えて，テンプレートの値からMediaWikiの内部リンクマークアップを除去し，
テキストに変換せよ（参考: マークアップ早見表）．

[URL]
https://nlp100.github.io/ja/ch03.html#27-内部リンクの除去

[Ref]
- regex
    - https://docs.python.org/ja/3/library/re.html?#regular-expression-syntax
        - 特殊文字（special characters）
            - \number
                - 同じ番号のグループの中身にマッチします。
                - グループは 1 から始まる番号をつけられます。
- ウィキペディアのマークアップ早見表
    - https://ja.wikipedia.org/wiki/Help:%E6%97%A9%E8%A6%8B%E8%A1%A8

[Usage]
python knock27.py
"""
import os
import re
import sys
from collections import OrderedDict

from knock26 import remove_em

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.pickle import load  # noqa: E402 isort:skip
from kiyuna.utils.message import Renderer, message  # noqa: E402 isort:skip
from kiyuna.utils.message import green  # noqa: E402 isort:skip


def remove_interlink(od: OrderedDict) -> OrderedDict:
    """remove interwiki link
        [[記事名]]             => 記事名
        [[記事名|表示文字]]     => 表示文字
        [[記事名#節名|表示文字]] => 表示文字
        以下は内部リンクマークアップではない
        [[ファイル:Royal Coat of Arms of the United Kingdom.svg|85px|イギリスの国章]]
    """
    res = OrderedDict()
    reg = re.compile(r"\[\[(?:[^:\]]+?\|)?([^:]+?)\]\]")
    for key in od:
        res[key] = reg.sub(r"\1", od[key])
    return res


if __name__ == "__main__":
    infobox = load("infobox")
    res = remove_em(infobox)
    res = remove_interlink(res)

    with Renderer("knock27") as out:
        for (key, src), (_, dst) in zip(infobox.items(), res.items()):
            if src == dst:
                out.cnt += 1
            else:
                out.result(key, (src, green(dst)))
        if infobox == res:
            message("変化なし", type="warning")
