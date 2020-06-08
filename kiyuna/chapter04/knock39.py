r"""knock39.py
39. Zipfの法則
単語の出現頻度順位を横軸，その出現頻度を縦軸として，両対数グラフをプロットせよ．

[URL]
https://nlp100.github.io/ja/ch04.html#39-zipfの法則

[Ref]
- Zipf's law
    - https://ja.wikipedia.org/wiki/ジップの法則
        - 詳細は，リンク先または，下記の NOTE を参照

[Usage]
python knock39.py
"""
import matplotlib.pyplot as plt

from knock35 import build_cnter

if __name__ == "__main__":
    query = {"surface": "表層形"}
    num = None
    xlim_max = 30

    cnter = build_cnter(query)
    _, data = zip(*cnter.most_common(num))

    plt.scatter(range(1, len(data) + 1), data, s=3)
    plt.title("Zipf の法則")
    plt.xlabel("単語の出現頻度順位")
    plt.ylabel("出現頻度")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("out39.png")


""" NOTE
出現頻度が k 番目に大きい要素が全体に占める割合が 1/k に比例するという経験則である。
Zipf は「ジフ」と読まれることもある。
また、この法則が機能する世界を「ジフ構造」と記する論者もいる。
包括的な理論的説明はまだ成功していないものの、様々な現象に適用できることが知られている。
この法則に従う確率分布（離散分布）をジップ分布という。
ジップ分布はゼータ分布（英語版）の特殊な形である。
- 由来
    元来は、アメリカの言語学者ジョージ・キングズリー・ジップが
    英語の単語の出現頻度とその順位に関して発見した言語学の法則である。
- 法則が成立する現象の例
次のような様々な現象（自然現象、社会現象など）に成り立つ場合があることが確認されている：
    - 単語の出現頻度：言語全体だけでなく、例えば「ハムレット」など
        1作品中でも成り立つことが示されている。
    - ウェブページへのアクセス頻度
    - 都市の人口（都市の順位・規模法則）
    - 上位3%の人々の収入
    - 音楽における音符の使用頻度
    - 細胞内での遺伝子の発現量
    - 地震の規模
    - 固体が割れたときの破片の大きさ
"""
