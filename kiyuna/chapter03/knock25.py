r"""knock25.py
25. テンプレートの抽出
記事中に含まれる「基礎情報」テンプレートのフィールド名と値を抽出し，
辞書オブジェクトとして格納せよ．

[URL]
https://nlp100.github.io/ja/ch03.html#25-テンプレートの抽出

[Ref]
- re.DOTALL / re.S
    - https://docs.python.org/ja/3/library/re.html#re.S
        - '.' 特殊文字を、改行を含むあらゆる文字にマッチさせます。
        - このフラグがなければ、'.' は、改行 以外の あらゆる文字とマッチします。
        - インラインフラグの (?s) に相当します。
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

from kiyuna.utils.message import (  # noqa: E402 isort:skip
    Renderer,
    green,
    message,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.pickle import dump, load  # noqa: E402 isort:skip


if __name__ == "__main__":
    wiki = load("UK")

    preptn = r"(?P<Annotation>注記 = .+?\n)"
    annotation = re.search(preptn, wiki).group("Annotation")

    ptn = rf"基礎情報 国\n(?P<Infobox_body>.+(?={annotation}))"
    match = re.search(ptn, wiki, re.DOTALL)
    info_box = "\n" + match.group("Infobox_body") + annotation

    subptn = r"(.+?)\s*=\s*(.+)"
    reg = re.compile(subptn, re.DOTALL)
    od = OrderedDict(
        reg.search(line.strip()).groups()
        for line in info_box.split("\n|")
        if line
    )

    dump(od, "infobox")

    with Renderer("knock25") as out:
        for k, v in od.items():
            out.result(k, green(v))
