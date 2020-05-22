r"""knock19.py
19. 各行の1コラム目の文字列の出現頻度を求め，出現頻度の高い順に並べる
各行の1列目の文字列の出現頻度を求め，その高い順に並べて表示せよ．
確認にはcut, uniq, sortコマンドを用いよ．

[URL]
https://nlp100.github.io/ja/ch02.html#19-各行の1コラム目の文字列の出現頻度を求め出現頻度の高い順に並べる

[Ref]
- sort と uniq でさくっとランキングを出力する
    - http://blog.nomadscafe.jp/2012/07/sort-uniq.html

[Command]
uniq
    -c 各行の前に出現回数を出力する

[Usage]
INPUT_PATH=./popular-names.txt
python knock19.py $INPUT_PATH
cut -f1 $INPUT_PATH | sort | uniq -c | sort -nr
diff -s <(python knock19.py $INPUT_PATH) <(cut -f1 $INPUT_PATH | sort | uniq -c | sort -nr)
"""
import os
import sys
from collections import Counter, defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer, message  # isort:skip


if __name__ == "__main__":
    path = sys.argv[1]

    cnter = defaultdict(int)
    with open(path) as f:
        for line in f:
            key, *_ = line.split("\t")
            cnter[key] += 1

    # cnter = Counter(line.split()[0] for line in open(path))

    for k, v in sorted(
        cnter.items(), key=lambda x: (x[1], x[0]), reverse=True
    ):
        print(f"{v:4d} {k}")
