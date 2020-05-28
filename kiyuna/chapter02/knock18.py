r"""knock18.py
18. 各行を3コラム目の数値の降順にソート
各行を3コラム目の数値の逆順で整列せよ（注意: 各行の内容は変更せずに並び替えよ）．
確認にはsortコマンドを用いよ（この問題はコマンドで実行した時の結果と合わなくてもよい）．

[URL]
https://nlp100.github.io/ja/ch02.html#18-各行を3コラム目の数値の降順にソート

[Command]
sort
    -k 場所と並べ替え種別を指定する
    -n 文字列を数値と見なして並べ替える
    -r 逆順で並べ替える
    -s stable

[Usage]
INPUT_PATH=./popular-names.txt
python knock18.py $INPUT_PATH > out18a
sort -k3nr $INPUT_PATH > out18b
diff -s <(python knock18.py $INPUT_PATH) <(sort -k3nr -s $INPUT_PATH)
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer, message  # noqa: E402 isort:skip


if __name__ == "__main__":
    path = sys.argv[1]

    with open(path) as f:
        lines = f.readlines()
        lines.sort(key=lambda line: -int(line.split()[2]))

    sys.stdout.writelines(lines)
