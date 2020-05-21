r"""knock09.py
09. Typoglycemia
スペースで区切られた単語列に対して，各単語の先頭と末尾の文字は残し，
それ以外の文字の順序をランダムに並び替えるプログラムを作成せよ．
ただし，長さが４以下の単語は並び替えないこととする．
適当な英語の文（例えば”I couldn’t believe that I could actually understand
what I was reading : the phenomenal power of the human mind .”）を与え，
その実行結果を確認せよ．

[URL]
https://nlp100.github.io/ja/ch01.html#09-typoglycemia

[Ref]
- タイポグリセミア
    - https://ja.wikipedia.org/wiki/タイポグリセミア

[Usage]
python knock09.py
"""
import os
import random
import string
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer, message  # isort:skip


if __name__ == "__main__":

    sent = (
        "I couldn't believe that I could actually understand"
        "what I was reading : the phenomenal power of the human mind ."
    )

    with Renderer("knock09") as out:
        res = []
        for word in sent.split(" "):
            if len(word) > 4:
                head, *middle, tail = word
                random.shuffle(middle)
                word = head + "".join(middle) + tail
            res.append(word)
        out.result("plain", sent)
        out.result("result", " ".join(res))
