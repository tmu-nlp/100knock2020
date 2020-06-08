r"""knock25.py
25. テンプレートの抽出
記事中に含まれる「基礎情報」テンプレートのフィールド名と値を抽出し，
辞書オブジェクトとして格納せよ．

[URL]
https://nlp100.github.io/ja/ch03.html#25-テンプレートの抽出

[Ref]
- re.DOTALL / re.S
    - https://docs.python.org/ja/3/library/re.html#re.S
        - '.' 特殊文字
            - フラグなし: 改行 以外の あらゆる文字とマッチします
            - フラグあり: 改行を含むあらゆる文字にマッチさせます
        - インラインフラグの (?s) に相当
- re.MULTILINE / re.M
    - https://docs.python.org/ja/3/library/re.html?#re.M
        - パターン文字 '^'
            - デフォルトでは、 文字列の先頭でのみマッチ
            - されていると、各行の先頭 (各改行の直後) でもマッチするようになる
            - https://docs.python.org/ja/3/library/re.html?#search-vs-match
        - パターン文字 '$'
            - デフォルトでは、 文字列の末尾および文字列の末尾の改行 (もしあれば) の直前
            - 指定されていると、文字列の末尾で、および各行の末尾 (各改行の直前)
        - インラインフラグの (?m) に相当
- regex
    - https://docs.python.org/ja/3/library/re.html?#regular-expression-syntax
        - 特殊文字（special characters）
            - (?=...)
                - 先読みアサーション (lookahead assertion)
                - ... が次に続くものにマッチすればマッチする
- ウィキペディアの「基礎情報 国」
    - https://ja.wikipedia.org/wiki/Template:基礎情報_国

[Usage]
python knock25.py
"""
import os
import re
import sys
from collections import OrderedDict
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.pickle import dump, load  # noqa: E402 isort:skip
from kiyuna.utils.message import Renderer, message  # noqa: E402 isort:skip
from kiyuna.utils.message import green  # noqa: E402 isort:skip


def parse(text: str, bracket: str = "{}", endl: str = "<br />") -> List[str]:
    res = []
    stack = []
    init = True
    begin, end = bracket
    for c in text:
        if c in bracket:
            if c == begin:
                res.append(" " * len(stack) + begin)
                stack.append(c)
            elif c == end:
                stack.pop(-1)
                res.append(" " * len(stack) + end)
            init = True
        else:
            if init:
                res.append(" " * len(stack))
                init = False
            res[-1] += endl if c == "\n" else c
    return res


def extract_infobox(wiki: str, endl: str = "\n") -> str:
    reg = re.compile(r"\s*")

    def get_level(line: str) -> int:
        return reg.match(line).end()

    wiki_parsed = parse(wiki, endl=endl)
    fst = next(i for i, line in enumerate(wiki_parsed) if "基礎情報" in line)
    lvl = get_level(wiki_parsed[fst])
    for lst in range(fst, len(wiki_parsed)):
        if get_level(wiki_parsed[lst]) < lvl:
            break
    return "".join(wiki_parsed[idx].lstrip(" ") for idx in range(fst, lst))


if __name__ == "__main__":
    wiki = load("UK")

    infobox = re.search(
        r"""
        ^{{基礎情報\s国
        (?P<Infobox_body>.+?)
        ^}}$
        """,
        wiki,
        flags=re.VERBOSE | re.DOTALL | re.MULTILINE,
    ).group("Infobox_body")
    reg = re.compile(r"(.+?)\s*=\s*(.+)", re.DOTALL)
    od = OrderedDict(
        reg.search(line.strip()).groups()
        for line in infobox.split("\n|")
        if line
    )
    dump(od, "infobox")

    with Renderer("knock25") as out:
        for k, v in od.items():
            out.result(k, green(v))

    assert od == OrderedDict(
        reg.search(line.strip()).groups()
        for line in extract_infobox(wiki).lstrip("基礎情報 国").split("\n|")
        if line
    )

    assert od == OrderedDict(
        re.findall(
            r"""
            \|                  # |
            (?P<Key>.+?)        # 略名
            \s*                 # _
            =                   # =
            \s*                 # _
            (?P<Value>.+?)      # イギリス
            (?=                 # \n| or \n$ が後ろに続く
                \n
                (?:
                    \|
                    |
                    $
                )
            )
            """,
            extract_infobox(wiki),
            flags=re.X | re.S | re.M,
        )
    )
