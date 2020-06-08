r"""knock31.py
31. 動詞
動詞の表層形をすべて抽出せよ．

[URL]
https://nlp100.github.io/ja/ch04.html#31-動詞

[Usage]
python knock31.py
"""
from knock30 import test_extract

if __name__ == "__main__":
    query = {
        "title": "knock31",
        "src": {"pos": "動詞"},
        "dst": {"surface": "表層形"},
    }
    test_extract(query, verbose=True)
