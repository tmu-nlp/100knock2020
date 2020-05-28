r"""knock15.py
15. 末尾のN行を出力
自然数Nをコマンドライン引数などの手段で受け取り，入力のうち末尾のN行だけを表示せよ．
確認にはtailコマンドを用いよ．

[URL]
https://nlp100.github.io/ja/ch02.html#15-末尾のn行を出力

[Command]
tail
    -c 末尾の指定したバイト数のみ表示
    -n 末尾の指定した行数のみ表示
    -q ファイルごとのヘッダ表示を行わない（複数ファイル指定時に使う）
    -v 常にファイルごとのヘッダ出力を行う
    -f ファイルを監視して内容が追加されるたびに末尾に表示（ログ監視などに使う）

[Usage]
INPUT_PATH=./popular-names.txt
N=4
python knock15.py $INPUT_PATH $N
diff -s <(tail -n $N $INPUT_PATH) <(python knock15.py $INPUT_PATH $N)
"""
import os
import sys
from collections import deque
from typing import Deque

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer, message  # noqa: E402 isort:skip


def tail(path: str, n: int) -> Deque[str]:
    with open(path) as f:
        return deque(f, n)


if __name__ == "__main__":
    path, n, *_ = sys.argv[1:]

    for line in tail(path, int(n)):
        sys.stdout.write(line)
