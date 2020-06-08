r"""knock38.py
38. ヒストグラム
単語の出現頻度のヒストグラム
（横軸に出現頻度，縦軸に出現頻度をとる単語の種類数を棒グラフで表したもの）を描け．

[URL]
https://nlp100.github.io/ja/ch04.html#38-ヒストグラム

[Usage]
python knock38.py
"""
import matplotlib.pyplot as plt

from knock35 import build_cnter

if __name__ == "__main__":
    query = {"surface": "表層形"}
    num = None
    xlim_max = 30

    cnter = build_cnter(query)
    _, data = zip(*cnter.most_common(num))

    plt.hist(data, bins=range(xlim_max + 1))
    plt.title("ヒストグラム")
    plt.xlabel("出現頻度")
    plt.ylabel("単語の種類数")
    plt.xlim([1, xlim_max])
    plt.savefig("out38.png")
