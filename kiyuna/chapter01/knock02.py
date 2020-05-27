r"""knock02.py
02. 「パトカー」＋「タクシー」＝「パタトクカシーー」
「パトカー」＋「タクシー」の文字を先頭から交互に連結して文字列「パタトクカシーー」を得よ．

[URL]
https://nlp100.github.io/ja/ch01.html#02-パトカータクシーパタトクカシーー

[Ref]
- zip_longest
    - https://docs.python.org/ja/3/library/itertools.html#itertools.zip_longest
- 引数リストのアンパック
    - https://docs.python.org/ja/3.6/tutorial/controlflow.html#unpacking-argument-lists

```
l = ['one', 'two', 'three']
print('{}-{}-{}'.format(*l))
# one-two-three

d = {'name': 'Alice', 'age': 20}
print('{name} is {age} years old.'.format(**d))
# Alice is 20 years old.
```

[Usage]
python knock02.py
"""
import os
import sys
from itertools import zip_longest
from typing import Iterator, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer, message  # noqa: E402 isort:skip


def concat(iterator: Iterator[Tuple[str]]) -> str:
    return "".join("".join(chars) for chars in iterator)


if __name__ == "__main__":

    with Renderer("knock02") as out:

        s, t = "パトカー", "タクシー"
        out.result([s, t], concat(zip(s, t)))

        s, t = "パトカー", "リムジンバス"
        out.result([s, t], concat(zip_longest(s, t, fillvalue="")))
        out.result([s, t], concat(zip_longest(s, t, fillvalue="＠")))

        ss = ["１２３", "ＡＢＣ", "ａｂｃ"]
        out.result(ss, concat(zip(*ss)))
