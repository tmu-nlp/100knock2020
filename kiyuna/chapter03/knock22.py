r"""knock22.py
22. カテゴリ名の抽出
記事のカテゴリ名を（行単位ではなく名前で）抽出せよ．

[URL]
https://nlp100.github.io/ja/ch03.html#22-カテゴリ名の抽出

[Ref]
- re.compile
    - https://docs.python.org/ja/3/library/re.html?#re.compile
        - re.compile() を使い、結果の正規表現オブジェクトを保存して再利用するほうが、
          一つのプログラムでその表現を何回も使うときに効率的です。
- re.search
    - https://docs.python.org/ja/3/library/re.html?#re.search
- マッチオブジェクト
    - https://docs.python.org/ja/3/library/re.html?#match-objects
- regex
    - https://docs.python.org/ja/3/library/re.html?#regular-expression-syntax
        - 特殊文字（special characters）
            - (?P<name>...)
            - +?
- ウィキペディアのカテゴリ
    - https://ja.wikipedia.org/wiki/Help:カテゴリ
        - [[Category:カテゴリ名]] or [[Category:カテゴリ名|ソートキー]]

[Usage]
python knock22.py
"""
import os
import re
import sys
from typing import Iterator, Match, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer, message  # noqa: E402 isort:skip
from kiyuna.utils.pickle import load  # noqa: E402 isort:skip


def exec_search(wiki: str, pattern: str) -> Iterator[Tuple[str, Match]]:
    reg = re.compile(pattern)
    for line in wiki.split("\n"):
        match = reg.search(line)
        if match:
            yield line, match


if __name__ == "__main__":
    wiki = load("UK")

    pat_category_only = r"\[\[Category:(?P<Category_name>.+?)(\||])"
    for _, match in exec_search(wiki, pat_category_only):
        print(match.group("Category_name"))

    pats = (
        pat_category_only,
        r"\[\[Category:(?P<Category_name>[^|]+)\|*(?P<Sortkey>.*)\]\]",
    )
    for pat in pats:
        with Renderer(pat) as out:
            for line, match in exec_search(wiki, pat):
                out.result(line, match.groups())
