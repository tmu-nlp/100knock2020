"""
44. 係り受け木の可視化
与えられた文の係り受け木を有向グラフとして可視化せよ．可視化には，Graphviz等を用いるとよい．
"""
from itertools import islice

import pydot

from knock41 import cabocha_into_chunks
from knock43 import ChunkNormalized

if __name__ == "__main__":

    def input_k():
        return int(input("Enter a number (0: exit) -> "))

    for k in iter(input_k, 0):
        for chunks in islice(cabocha_into_chunks(), k - 1, k):
            chunks = {k: ChunkNormalized(v) for k, v in chunks.items()}
            G = pydot.Dot(graph_type="digraph")
            for i, c in chunks.items():
                if c.norm:
                    color = "red" if c.has_pos("名詞") else "black"
                    G.add_node(pydot.Node(i, label=c.norm, color=color))
            for i, c in chunks.items():
                if c.dst == -1:
                    continue
                if c.norm and c.dst in chunks and chunks[c.dst].norm:
                    G.add_edge(pydot.Edge(i, c.dst))
            G.write_png(f"./out44_{k}.png")
