r"""knock12.py
12. 1列目をcol1.txtに，2列目をcol2.txtに保存
各行の1列目だけを抜き出したものをcol1.txtに，
2列目だけを抜き出したものをcol2.txtとしてファイルに保存せよ．
確認にはcutコマンドを用いよ．

[URL]
https://nlp100.github.io/ja/ch02.html#12-1列目をcol1txtに2列目をcol2txtに保存

[Ref]
- ファイルがあるかチェックする方法
    - http://sweng.web.fc2.com/ja/program/bash/bash-check-file.html

[Command]
cut
    -b 必要な項目をバイト数で指定する
    -d 区切り文字を指定する
    -f 必要な項目を項目数で指定する

[Usage]
INPUT_PATH=./popular-names.txt
python knock12.py $INPUT_PATH
cut -f1-2 $INPUT_PATH
if [ -e col1.txt ]; then
    diff -s col1.txt <(cut -f 1 $INPUT_PATH)
fi
if [ -e col2.txt ]; then
    diff -s col2.txt <(cut -f 2 $INPUT_PATH)
fi
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer, message  # noqa: E402 isort:skip


input_path = "popular-names.txt"
col1, col2 = "col1.txt", "col2.txt"

with open(col1, "w") as f1, open(col2, "w") as f2, open(input_path) as f_in:
    for line in f_in:
        cols = line.split("\t")
        f1.write(cols[0] + "\n")
        print(cols[1], file=f2)

for i in range(4):
    with open(f"col{i + 1}.txt", "w") as f:
        f.writelines(line.split()[i] + "\n" for line in open(input_path))
