r"""knock04.py
04. 元素記号
“Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also
Sign Peace Security Clause. Arthur King Can.”という文を単語に分解し，
1, 5, 6, 7, 8, 9, 15, 16, 19番目の単語は先頭の1文字，
それ以外の単語は先頭に2文字を取り出し，
取り出した文字列から単語の位置（先頭から何番目の単語か）への連想配列（辞書型もしくはマップ型）を
作成せよ．

[URL]
https://nlp100.github.io/ja/ch01.html#04-元素記号

[Usage]
python knock04.py
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer, message  # noqa: E402 isort:skip


if __name__ == "__main__":

    s = "Hi He Lied Because Boron Could Not Oxidize Fluorine.\
         New Nations Might Also Sign Peace Security Clause. Arthur King Can."

    single_idxs = {1, 5, 6, 7, 8, 9, 15, 16, 19}

    with Renderer("knock04") as out:

        words = s.replace(".", "").split()

        res = {}
        for idx, word in enumerate(words, start=1):
            if idx in single_idxs:
                res[word[0:1]] = idx
            else:
                res[word[0:2]] = idx

        out.result("dict extension", res)

        res = {
            word[: 2 - (idx in single_idxs)]: idx
            for idx, word in enumerate(words, start=1)
        }

        out.result("dict comprehension", res)
