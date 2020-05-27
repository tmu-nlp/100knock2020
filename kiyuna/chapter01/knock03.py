r"""knock03.py
03. 円周率
“Now I need a drink, alcoholic of course, after the heavy lectures involving
quantum mechanics.”という文を単語に分解し，各単語の（アルファベットの）文字数を先頭から
出現順に並べたリストを作成せよ．

[URL]
https://nlp100.github.io/ja/ch01.html#03-円周率

[Ref]
- リスト内包表記
    - https://docs.python.org/ja/3/tutorial/datastructures.html#list-comprehensions
- rstrip
    - https://docs.python.org/ja/3/library/stdtypes.html#str.rstrip
- regex
    - https://docs.python.org/ja/3/library/re.html

[Tips]
>>> 'mississippi'.rstrip('ipz')
'mississ'
>>> import re
>>> re.split(r'(\W+)', '...words, words...')
['', '...', 'words', ', ', 'words', '...', '']

[Usage]
python knock03.py
"""
import doctest
import os
import re
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer, message  # noqa: E402 isort:skip


if __name__ == "__main__":
    doctest.testmod(verbose=True)

    s = (
        "Now I need a drink, alcoholic of course, "
        "after the heavy lectures involving quantum mechanics."
    )

    with Renderer("knock03") as out:
        out.result(
            "replace + list comprehension",
            [len(w) for w in s.replace(",", "").replace(".", "").split()],
        )
        out.result(
            "rstrip + list comprehension",
            [len(w.rstrip(",.")) for w in s.split()],
        )
        out.result("re.findall + map", list(map(len, re.findall(r"\w+", s))))
        out.result("re.sub + map", [*map(len, re.sub(r"[,.]", "", s).split())])
        out.result("re.split + map", [*map(len, re.split(r"\W+", s)[:-1])])
