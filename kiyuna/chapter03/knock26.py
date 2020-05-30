r"""knock26.py
26. 強調マークアップの除去
25の処理時に，テンプレートの値からMediaWikiの強調マークアップ
（弱い強調，強調，強い強調のすべて）を除去してテキストに変換せよ（参考: マークアップ早見表）．

[URL]
https://nlp100.github.io/ja/ch03.html#26-強調マークアップの除去

[Ref]
- re.sub
    - https://docs.python.org/ja/3/library/re.html#re.sub
- regex
    - https://docs.python.org/ja/3/library/re.html?#regular-expression-syntax
        - 特殊文字（special characters）
            - {m,n}
- ウィキペディアのマークアップ早見表
    - https://ja.wikipedia.org/wiki/Help:%E6%97%A9%E8%A6%8B%E8%A1%A8

[Usage]
python knock26.py
"""
import os
import re
import sys
from collections import OrderedDict

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.pickle import load  # noqa: E402 isort:skip
from kiyuna.utils.message import Renderer, message  # noqa: E402 isort:skip
from kiyuna.utils.message import green  # noqa: E402 isort:skip


def remove_em(od: OrderedDict) -> OrderedDict:
    """remove emphasis expressions
        ''italics''
        '''bold'''
        '''''both'''''
    """
    res = OrderedDict()
    reg = re.compile(r"'{2,}")
    for key in od:
        res[key] = reg.sub("", od[key])
    return res


if __name__ == "__main__":
    infobox = load("infobox")
    res = remove_em(infobox)

    with Renderer("knock26") as out:
        for (key, src), (_, dst) in zip(infobox.items(), res.items()):
            if src == dst:
                out.cnt += 1
            else:
                out.result(key, (src, green(dst)))
        if infobox == res:
            message("変化なし", type="warning")
