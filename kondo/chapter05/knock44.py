"""
44. 係り受け木の可視化Permalink
与えられた文の係り受け木を有向グラフとして可視化せよ．
可視化には，Graphviz等を用いるとよい．
"""

from graphviz import Digraph

data = "ai.ja.txt.parsed"

class Morph:
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1

    def __str__(self):
        return "surface[{}]\tbase[{}]\tpos[{}]\tpos1[{}]".format(self.surface, self.base, self.pos, self.pos1)

class Chunk:
    def __init__(self):
        self.morphs = []
        self.dst = []
        self.srcs = []

    def __str__(self):
        surface = ""
        for morph in self.morphs:
            surface += morph.surface
        return "{}\tdst[{}]".format(surface, self.dst)

    def normalized_surface(self):
        #表層系の記号を除去する
        normalized_word = ""
        for morph in self.morphs:
            if morph.pos != "記号":
                normalized_word += morph.surface
        return normalized_word

    def check_pos(self, pos):
        for morph in self.morphs:
            if morph.pos == pos:
                return 1
        return 0

def chunk_to_list():
    with open(data, encoding = 'utf-8') as cabocha:
        chunks = {}
        for line in cabocha:
            if line == "EOS\n":
                if len(chunks) > 0:
                    sorted_chunks = sorted(chunks.items(), key = lambda x: x[0])
                    #[(key1, value1), (key2, value2)...]→[(key1, key2...), (value1, value2)]
                    #リストの中身を結合するので*
                    #valueを返す
                    yield list(zip(*sorted_chunks))[1]
                    chunks = {}
            elif line[0] == '*':
                cols = line.split()
                num = int(cols[1])
                #末尾のDを読まない
                dst = int(cols[2][:-1])
                chunks[num] = Chunk()
                chunks[num].dst = dst
            else:
                surface, els = line.split('\t')
                els = els.split(',')

                chunks[num].morphs.append(Morph(surface, els[6], els[0], els[1]))

        #raise StopIterationはRunTimeEror出る(python3.7以降)
        return

if __name__ == "__main__":
    #改行単位
    n = input("可視化したいのは何目？-->")
    n = int(n)
    for i, chunks in enumerate(chunk_to_list()):
        if i == n:
            dg = Digraph(format="png")
            dg.attr('node')
            for chunk in chunks:
                if chunk.dst != -1:
                    src = chunk.normalized_surface()
                    dst = chunks[chunk.dst].normalized_surface()
                    if src != "" and dst != "":
                        dg.node(src)
                        dg.node(dst)
                        dg.edge(src, dst)
            print(dg)
            dg.render('graph44')
            break