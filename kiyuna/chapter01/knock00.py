r"""knock00.py
00. 文字列の逆順
文字列”stressed”の文字を逆に（末尾から先頭に向かって）並べた文字列を得よ．

[URL]
https://nlp100.github.io/ja/ch01.html#00-文字列の逆順

[Ref]
- reversed
    - https://docs.python.org/ja/3/library/functions.html#reversed
- string は変更不能なシーケンス型
    - https://docs.python.org/ja/3/reference/datamodel.html

[Usage]
python knock00.py
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer, message  # isort:skip


if __name__ == "__main__":

    s = "stressed"

    with Renderer("knock00") as out:
        out.result("slice", s[::-1])
        out.result("reversed", "".join(reversed(s)))

    with Renderer("MEMO") as out:
        out.result("reversed の返り値は", reversed(s))
        try:
            s.reverse()
        except AttributeError as e:
            out.result("string は変更不能なシーケンス型（immutable sequence）", e)
