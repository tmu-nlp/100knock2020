r"""knock10.py
10. 行数のカウント
行数をカウントせよ．確認にはwcコマンドを用いよ．

[URL]
https://nlp100.github.io/ja/ch02.html#10-行数のカウント

[Ref]
- ジェネレータを使ってメモリを節約
    - https://www.python.org/dev/peps/pep-0289/

[Command]
wc (word count)
    -c バイト数を表示
    -l 改行の数を表示する
    -m 文字数を表示する（マルチバイト文字に対応）
    -w 単語数を表示する

[Usage]
INPUT_PATH=./popular-names.txt
python knock10.py $INPUT_PATH
wc -l $INPUT_PATH
cat $INPUT_PATH | wc -l
diff -sw <(python knock10.py $INPUT_PATH) <(cat $INPUT_PATH | wc -l)
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer, message  # noqa: E402 isort:skip


if __name__ == "__main__":
    path = sys.argv[1]

    with Renderer("knock10") as out, open(path) as f:
        out.result("generator", sum(1 for _ in f))
        out.result("readlines", len(open(path).readlines()))
        out.result("read", len(open(path).read().rstrip("\n").split("\n")))

    print(sum(1 for _ in open(path)))
