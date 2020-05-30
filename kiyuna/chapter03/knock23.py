r"""knock23.py
23. セクション構造
記事中に含まれるセクション名とそのレベル（例えば”== セクション名 ==”なら1）を表示せよ．

[URL]
https://nlp100.github.io/ja/ch03.html#23-セクション構造

[Ref]
- re.compile
    - https://docs.python.org/ja/3/library/re.html?#re.compile
        - re.compile() を使い、結果の正規表現オブジェクトを保存して再利用するほうが、
          一つのプログラムでその表現を何回も使うときに効率的です。
- re.search() vs. re.match()
    - https://docs.python.org/ja/3/library/re.html#search-vs-match
        - re.match()
            - 文字列の先頭でのみのマッチを確認する
        - re.search()
            - 文字列中の位置にかかわらずマッチを確認する
- マッチオブジェクト
    - https://docs.python.org/ja/3/library/re.html?#match-objects
- regex
    - https://docs.python.org/ja/3/library/re.html?#regular-expression-syntax
        - 特殊文字（special characters）
            - (?P=name)
                - 名前付きグループへの後方参照
            - ^
                - (キャレット) 文字列の先頭にマッチする
        - 特殊シーケンス（special sequences）
            - \s
                - Unicode 空白文字 (これは [ \t\n\r\f\v] その他多くの文字) にマッチする
- ウィキペディアのセクション
    - https://ja.wikipedia.org/wiki/Help:セクション

[Usage]
python knock23.py
"""
import os
import re
import sys
from typing import Iterator, List, Match, Tuple

from knock22 import exec_search

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer, message  # noqa: E402 isort:skip
from kiyuna.utils.pickle import load  # noqa: E402 isort:skip


def exec_match(wiki: List[str], pattern: str) -> Iterator[Tuple[str, Match]]:
    reg = re.compile(pattern)
    for line in wiki:
        match = reg.match(line)
        if match:
            yield line, match


if __name__ == "__main__":
    wiki = load("UK").split("\n")

    pat = r"(?P<Level>=+)\s*(?P<Heading>.+)\s*(?P=Level)"
    for _, match in exec_match(wiki, pat):
        level, heading = match.group(1, 2)
        print((heading, len(level) - 1))

    with Renderer("match") as out:
        for line, match in exec_match(wiki, pat):
            out.result(line, match.groups())

    pat_beginning = r"^" + pat
    with Renderer("search") as out:
        for line, match in exec_search(wiki, pat_beginning):
            out.result(line, match.groups())
