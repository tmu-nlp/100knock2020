r"""knock08.py
08. 暗号文
与えられた文字列の各文字を，以下の仕様で変換する関数cipherを実装せよ．

- 英小文字ならば(219 - 文字コード)の文字に置換
- その他の文字はそのまま出力

この関数を用い，英語のメッセージを暗号化・復号化せよ．

[URL]
https://nlp100.github.io/ja/ch01.html#08-暗号文

[Ref]
- ROT13
    - https://ja.wikipedia.org/wiki/ROT13

[Usage]
python knock08.py
"""
import os
import string
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer, message  # isort:skip


def rot13(text: str) -> str:
    return "".join(chr(219 - ord(c)) if c.islower() else c for c in text)


if __name__ == "__main__":

    d = {
        "printable": string.printable.strip(),
        "ascii_lowercase": string.ascii_lowercase,
    }

    for title, s in d.items():
        with Renderer(title) as out:
            out.result("plaintext", s)
            out.header("encode")
            out.result("ciphertext", rot13(s))
            out.header("decode")
            out.result("plaintext", rot13(rot13(s)))
