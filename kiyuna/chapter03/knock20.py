r"""knock20.py
20. JSONデータの読み込み
Wikipedia記事のJSONファイルを読み込み，「イギリス」に関する記事本文を表示せよ．

[URL]
https://nlp100.github.io/ja/ch03.html#20-jsonデータの読み込み

[Ref]
- gzip
    - https://docs.python.org/ja/3/library/gzip.html
        - gzip.open() の引数 mode に 'rt' を指定することでテキストモードになる
- json
    - https://docs.python.org/ja/3/library/json.html
        - json.load() は fp を Python オブジェクトへ脱直列化する
            - fp: .read() をサポートし JSON ドキュメントを含んでいる text/binary file
        - json.loads() は s を Python オブジェクトへ脱直列化する
            - s: JSON ドキュメントを含んでいる str/bytes/bytearray のインスタンス
- Wikipedia記事のイギリス
    - https://ja.wikipedia.org/wiki/イギリス
- ウィキペディアのスタイル
    - https://ja.wikipedia.org/wiki/Wikipedia:スタイルマニュアル
- ウィキリンク
    - https://ja.wikipedia.org/wiki/Help:リンク
        - ウィキリンク（または内部リンク）は、日本語版ウィキペディア内にある別のページへリンクします。
        - ウィキリンクを表記するには、リンクする先のページ名を2連の角括弧 [[ ]] で囲みます。

[Usage]
python knock20.py jawiki-country.json.gz イギリス UK
"""
import gzip
import json
import os
import pprint
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer, message  # noqa: E402 isort:skip
from kiyuna.utils.pickle import dump  # noqa: E402 isort:skip


def extract_wiki(path: str, title: str) -> dict:
    with gzip.open(path, "rt") as f:
        for line in f:
            d = json.loads(line)
            if d["title"] == title:
                return d


if __name__ == "__main__":
    input_path, title, out_fname, *_ = sys.argv[1:]

    article = extract_wiki(input_path, title)

    dump(article["text"], out_fname)

    with Renderer("MEMO") as out:
        message(article.keys())
        # pprint.pprint(article)
