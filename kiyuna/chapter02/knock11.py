r"""knock11.py
11-タブをスペースに置換
タブ1文字につきスペース1文字に置換せよ．
確認にはsedコマンド，trコマンド，もしくはexpandコマンドを用いよ．

[URL]
https://nlp100.github.io/ja/ch02.html#11-タブをスペースに置換

[Ref]
- BSD sed でタブを入力
    - http://mattintosh.hatenablog.com/entry/2013/01/16/143323

[Command]
sed (stream editor)
tr (translate)

[Usage]
INPUT_PATH=./popular-names.txt
python knock11.py $INPUT_PATH
# sed
sed $'s/\t/ /g' $INPUT_PATH
diff -sw <(python knock11.py $INPUT_PATH) <(sed $'s/\t/ /g' $INPUT_PATH)
# tr
tr '\t' ' ' < $INPUT_PATH
diff -sw <(python knock11.py $INPUT_PATH) <(tr '\t' ' ' < $INPUT_PATH)
# expand
expand -t 1 $INPUT_PATH
diff -sw <(python knock11.py $INPUT_PATH) <(expand -t 1 $INPUT_PATH)
"""
import os
import sys
from typing import Generator, TypeVar

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer, message  # noqa: E402 isort:skip

Path = TypeVar("Path", bound=str)


def tab2space(path: Path) -> Generator[str]:
    with open(path) as f:
        for line in f:
            yield line.replace("\t", " ")


if __name__ == "__main__":
    path: Path = sys.argv[1]

    sys.stdout.writelines(tab2space(path))
