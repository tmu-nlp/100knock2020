r"""knock24.py
24. ファイル参照の抽出
記事から参照されているメディアファイルをすべて抜き出せ．

[URL]
https://nlp100.github.io/ja/ch03.html#24-ファイル参照の抽出

[Ref]
- re.findall
    - https://docs.python.org/ja/3/library/re.html#re.findall
- re.IGNORECASE / re.I
    - https://docs.python.org/ja/3/library/re.html#re.I
        - 大文字・小文字を区別しないマッチングを行います;
        - [A-Z] のような正規表現は小文字にもマッチします。
        - インラインフラグの (?i) に相当します。
- regex
    - https://docs.python.org/ja/3/library/re.html?#regular-expression-syntax
        - 特殊文字（special characters）
            - (?:...)
                - 普通の丸括弧の、キャプチャしない版です。
- ウィキペディアの画像
    - https://ja.wikipedia.org/wiki/Help:画像の表示#要点
        - 書式：[[ファイル:ファイル名|オプション]]
        - プレフィックスは File でも機能します（[[File:ファイル名|オプション]]）。
- ウィキペディアの <gallery>
    - https://en.wikipedia.org/wiki/Help:Gallery_tag
- ウィキペディアでアップロード可能なファイル形式
    - https://ja.wikipedia.org/wiki/Help:画像などのファイルのアップロードと利用
        - ウィキペディアでは画像などのファイルをアップロードし、
          そのファイルをページ中に挿入することができる
        - ウィキメディア・プロジェクトで許可されているファイル形式は、
          拡張子が png, gif, jpg, jpeg, xcf, pdf, mid, ogg, svg, djvu のいずれか

[Usage]
python knock24.py
"""
import os
import re
import sys
from typing import Iterator, List, Match, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.pickle import load  # noqa: E402 isort:skip
from kiyuna.utils.message import Renderer, message  # noqa: E402 isort:skip
from kiyuna.utils.message import trunc, green  # noqa: E402 isort:skip


def exec_findall(wiki: List[str], pattern: str) -> Iterator[Tuple[str, Match]]:
    reg = re.compile(pattern)
    for line in wiki:
        for match in reg.findall(line):  # 1 行に複数あっても大丈夫
            yield line, match


if __name__ == "__main__":
    wiki = load("UK").split("\n")

    pat = (
        r"(?:\s=\s)?"  # 「基礎情報 国]」対策
        r"([^:=]+)"  # '/' を [^] の中に追加すると <ref> 内のファイル名も取得できる
        r"\.(?i)(png|gif|jpg|jpeg|xcf|pdf|mid|ogg|svg|djvu)"
    )
    with Renderer("knock24") as out:
        for line, filename in exec_findall(wiki, pat):
            fname = ".".join(filename)
            if "/" not in fname:  # <ref> 対策
                out.result(trunc(line), green(fname))

    """ NOTE
    - ウィキペディアの画像
        - [[ファイル:Uk topo en.jpg|thumb|200px|イギリスの地形図]]
    - 基礎情報 国
        - |国旗画像 = Flag of the United Kingdom.svg
    - <gallery>
        - Stonehenge2007 07 30.jpg|[[ストーンヘンジ]]
    - <ref>
        - <ref>[http://warp.da.ndl.go.jp/.../country.pdf
    """
