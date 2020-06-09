r"""knock21.py
21. カテゴリ名を含む行を抽出
記事中でカテゴリ名を宣言している行を抽出せよ．

[URL]
https://nlp100.github.io/ja/ch03.html#21-カテゴリ名を含む行を抽出

[Ref]
- re.compile
    - https://docs.python.org/ja/3/library/re.html?#re.compile
        - re.compile() を使い、結果の正規表現オブジェクトを保存して再利用するほうが、
          一つのプログラムでその表現を何回も使うときに効率的です。
- re.fullmatch
    - https://docs.python.org/ja/3/library/re.html?#re.fullmatch
- ウィキペディアのカテゴリ
    - https://ja.wikipedia.org/wiki/Help:カテゴリ
        - [[Category:カテゴリ名]] or [[Category:カテゴリ名|ソートキー]]

[Usage]
python knock21.py
"""
import os
import re
import sys
from typing import Iterator

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer, message  # noqa: E402 isort:skip
from kiyuna.utils.pickle import load  # noqa: E402 isort:skip


def exec_fullmatch(wiki: str, pattern: str) -> Iterator[str]:
    reg = re.compile(pattern)
    for line in wiki.split("\n"):
        if reg.fullmatch(line):
            yield line


if __name__ == "__main__":
    wiki = load("UK")

    for category in exec_fullmatch(wiki, r"\[\[Category:.+\]\]"):
        print(category)

    with Renderer("MEMO") as out:
        lines = tuple(line for line in wiki.split("\n") if "Category" in line)
        out.result("Category を含む行", lines)
