r"""knock14.py
14. 先頭からN行を出力
自然数Nをコマンドライン引数などの手段で受け取り，入力のうち先頭のN行だけを表示せよ．
確認にはheadコマンドを用いよ．

[URL]
https://nlp100.github.io/ja/ch02.html#14-先頭からn行を出力

[Command]
head
    -c 先頭から指定したバイト数のみ表示
    -n 先頭から指定した行数のみ表示する
    -q ファイルごとのヘッダ表示を行わない（複数ファイル指定時に使う）
    -v 常にファイルごとのヘッダ出力を行う

[Usage]
INPUT_PATH=./popular-names.txt
N=4
python knock14.py $INPUT_PATH $N
diff -s <(head -n $N $INPUT_PATH) <(python knock14.py $INPUT_PATH $N)
"""
import os
import sys
from typing import Iterator

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer, message  # noqa: E402 isort:skip


def head(path: str, n: int) -> Iterator[str]:
    with open(path) as f:
        for _ in range(n):
            yield f.readline()


if __name__ == "__main__":
    path, n, *_ = sys.argv[1:]

    for line in head(path, int(n)):
        sys.stdout.write(line)
