r"""knock36.py
36. 頻度上位10語
出現頻度が高い10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ．

[URL]
https://nlp100.github.io/ja/ch04.html#36-頻度上位10語

[Usage]
python knock36.py
"""
import matplotlib.pyplot as plt

from knock35 import build_cnter

if __name__ == "__main__":
    query = {"surface": "表層形"}
    num = 10

    cnter = build_cnter(query)
    labels, data = zip(*cnter.most_common(num))

    plt.bar(range(num), data)
    plt.title(f"頻度上位 {num} 語")
    plt.xticks(range(num), map(lambda x: f'"{x}"', labels))
    plt.xlabel(f"単語（{query}）")
    plt.ylabel("出現頻度")
    plt.savefig("out36.png")
