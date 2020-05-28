r"""knock07.py
07. テンプレートによる文生成
引数x, y, zを受け取り「x時のyはz」という文字列を返す関数を実装せよ．
さらに，x=12, y=”気温”, z=22.4として，実行結果を確認せよ．

[URL]
https://nlp100.github.io/ja/ch01.html#07-テンプレートによる文生成

[Ref]
- フォーマット
    - https://docs.python.org/ja/3/tutorial/inputoutput.html

[Usage]
python knock07.py
"""
import os
import string
import sys
from typing import Iterator

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer, message  # noqa: E402 isort:skip


def knock07(x: int, y: str, z: float) -> Iterator[str]:

    yield "f-strings", f"{x}時の{y}は{z}"

    yield "string.format", "{}時の{}は{}".format(x, y, z)

    format_spec = "${hour}時の${info}は${value}"
    template = string.Template(format_spec)
    yield "string.Template", template.substitute(hour=x, info=y, value=z)

    yield "printf-style", "%d時の%sは%.1f" % (x, y, z)


if __name__ == "__main__":

    x, y, z = 12, "気温", 22.4

    with Renderer("knock07") as out:
        for head, body in knock07(x, y, z):
            out.result(head, body)
