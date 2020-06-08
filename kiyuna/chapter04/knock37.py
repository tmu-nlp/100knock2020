r"""knock37.py
37. 「猫」と共起頻度の高い上位10語 :new:
「猫」とよく共起する（共起頻度が高い）10語とその出現頻度を
グラフ（例えば棒グラフなど）で表示せよ．

[URL]
https://nlp100.github.io/ja/ch04.html#37-猫と共起頻度の高い上位10語

[Usage]
python knock37.py
"""
from collections import Counter
from typing import Dict, List

import matplotlib.pyplot as plt

from knock30 import mecab_into_sentences
from knock35 import build_cnter

Morpheme = Dict[str, str]
Sentence = List[Morpheme]


if __name__ == "__main__":
    tgt_key, tgt_val = "surface", "猫"
    num = 10
    exclude_pos = {"助詞", "助動詞", "記号"}

    cnter = Counter()
    for sentence in mecab_into_sentences():
        if any(d[tgt_key] == tgt_val for d in sentence):
            cnter += Counter(
                d[tgt_key] for d in sentence  # if d["pos"] not in exclude_pos
            )
    del cnter[tgt_val]
    for k, v in build_cnter({"surface": "表層形"}).most_common(50):
        del cnter[k]
    labels, data = zip(*cnter.most_common(num))

    plt.bar(range(num), data)
    plt.title(f"「{tgt_val}」と共起頻度の高い上位 {num} 語")
    plt.xticks(range(num), map(lambda x: f'"{x}"', labels))
    plt.xlabel(f"単語（{tgt_key}）")
    plt.ylabel("出現頻度")
    plt.savefig("out37.png")
