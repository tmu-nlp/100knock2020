r"""knock32.py
32. 動詞の原形
動詞の原形をすべて抽出せよ．

[URL]
https://nlp100.github.io/ja/ch04.html#32-動詞の原形

[NOTE]
- サ変接続の名詞
    - 「見当する」-> 見当をつける/検討する

[Usage]
python knock32.py
"""
from knock30 import test_extract

if __name__ == "__main__":
    query = {
        "title": "knock32",
        "src": {"pos": "動詞"},
        "dst": {"base": "原形"},
    }
    test_extract(query, verbose=True)

    query = {
        "title": "もともと knock33 だったもの",
        "src": {"pos1": "サ変接続"},
        "dst": {"surface": "名詞"},
    }
    test_extract(query, verbose=True)
