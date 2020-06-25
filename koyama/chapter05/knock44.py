# 44. 係り受け木の可視化
# 与えられた文の係り受け木を有向グラフとして可視化せよ．
# 可視化には，Graphviz等を用いるとよい．

from knock41 import getdata
from graphviz import Digraph

if __name__ == "__main__":
    cabocha_file_path = "ai.ja.split.txt.parsed"
    document = getdata(cabocha_file_path)
    for i, sentence in enumerate(document):
        G = Digraph(format="png")
        G.attr("node", shape="circle")
        N = len(sentence)
        for chunk in sentence:
            src = ""
            for morph in chunk.morphs:
                src += morph.surface
            G.node(src, src)
            trg = ""
            if chunk.dst == -1:
                break
            for morph in sentence[chunk.dst].morphs:
                trg += morph.surface
            G.edge(src, trg)
        G.render(f"result44_{i}")
