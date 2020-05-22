r"""knock17.py
17. １列目の文字列の異なり
1列目の文字列の種類（異なる文字列の集合）を求めよ．確認にはcut, sort, uniqコマンドを用いよ．

[URL]
https://nlp100.github.io/ja/ch02.html#17-１列目の文字列の異なり

[Tips]
- ソート順に不満がある場合はロケール環境変数を設定する
    - LC_ALL=C
        一時的に英語ロケールでプログラムを実行したい場合に、
        コマンドの頭に「LC_ALL=C」をつけて実行する

[Command]
uniq
    連続した行をまとめるだけ．sort が必要

[Usage]
INPUT_PATH=./popular-names.txt
python knock17.py $INPUT_PATH
cut -f1 $INPUT_PATH | LC_ALL=C sort | uniq
diff -s <(python knock17.py $INPUT_PATH) <(cut -f1 $INPUT_PATH | sort | uniq)
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer, message  # isort:skip


if __name__ == "__main__":
    path = sys.argv[1]

    s = set()
    with open(path) as f:
        for line in f:
            cols = line.split("\t")
            s.add(cols[0])
    for e in sorted(s):
        print(e)
