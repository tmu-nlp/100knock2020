r"""knock13.py
13. col1.txtとcol2.txtをマージ
12で作ったcol1.txtとcol2.txtを結合し，
元のファイルの1列目と2列目をタブ区切りで並べたテキストファイルを作成せよ．
確認にはpasteコマンドを用いよ．

[URL]
https://nlp100.github.io/ja/ch02.html#13-col1txtとcol2txtをマージ

[Ref]
- ファイルディスクリプタ
    - https://qiita.com/ueokande/items/c75de7c9df2bcceda7a9
    - https://qiita.com/laikuaut/items/e1cc312ffc7ec2c872fc
- 標準入力同士の diff
    - https://qiita.com/bsdhack/items/55d5eced2fb3e6625d74

[Command]
paste
    -d 区切り文字を指定する
    -s ファイル単位で連結
cat
    FD0 を FD1 に流すコマンド

[Usage]
INPUT_PATH=./popular-names.txt
python knock13.py
# diff の取り方 3 種類
paste -d $'\t' col1.txt col2.txt | diff -s out13a -
cat out13a | (paste col1.txt col2.txt | diff -s /dev/fd/3 -) 3<&0
if ls out13b; then
    diff -s out13b $INPUT_PATH
fi
"""
import os
import sys
from typing import Generator

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer, message  # noqa: E402 isort:skip


def merge_cols(*fnames: str, sep=" ") -> Generator[str]:
    for cols in zip(*map(open, fnames)):
        yield sep.join(map(lambda col: col.rstrip(), cols))


if __name__ == "__main__":

    if any(not os.path.exists(f"col{i}.txt") for i in range(1, 5)):
        import knock12

    with open("out13a", "w") as f:
        for line in merge_cols("col1.txt", "col2.txt", sep="\t"):
            f.write(line + "\n")

    fnames = [f"col{i}.txt" for i in range(1, 5)]
    with open("out13b", "w") as f:
        f.writelines(line + "\n" for line in merge_cols(*fnames, sep="\t"))
