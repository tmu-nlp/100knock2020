r"""knock01.py
01. 「パタトクカシーー」
「パタトクカシーー」という文字列の1,3,5,7文字目を取り出して連結した文字列を得よ．

[URL]
https://nlp100.github.io/ja/ch01.html#01-パタトクカシーー

[Ref]
- スライス
    - https://docs.python.org/ja/3/glossary.html#term-slice

[Usage]
python knock01.py
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer, message  # noqa: E402 isort:skip


if __name__ == "__main__":

    s = "パタトクカシーー"

    with Renderer("knock01") as out:
        out.result("slice: 1, 3, 5, 7", s[::2])
        out.result("slice: 2, 4, 6, 8", s[1::2])
