'''
44. 係り受け木の可視化Permalink
与えられた文の係り受け木を有向グラフとして可視化せよ．可視化には，Graphviz等を用いるとよい．
'''
#表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音
import CaboCha
import re
import io
import pydot
font = 'IPAGothic'

class Morph:
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1

    def print(self):
        print("[surface] : {}\t[base] : {}\t[pos] : {}\t[pos1] : {}".format(\
            self.surface, self.base, self.pos, self.pos1))


class Chunk:
    def __init__(self):
        self.morphs = []
        self.dst = -1
        self.srcs = []
        self.idx = -1

    def make(self):
        chunk = ''
        for morph in self.morphs:
            # 記号を除去
            if morph.pos == "記号": continue
            chunk += morph.surface
        self.chunk = chunk

    def print(self):
        print("[文節番号:{}] {}\t[係り先:{}]\t[係り元{}]".format(self.idx, self.chunk\
                ,self.dst, self.srcs))

# 与えられた文の解析結果をラティスで出力
def parse_sentence(raw_sentence):
    format = CaboCha.FORMAT_LATTICE
    cabocha = CaboCha.Parser("-d /var/lib/mecab/dic/ipadic-utf8")
    parsed_sentence = cabocha.parse(raw_sentence)
    return parsed_sentence.toString(format)


def make_sentence(text):
    file = io.StringIO(text)
    chunks = {}
    for line in file:
        if line == 'EOS\n':
            chunks = extract_values(chunks)
            sentence = chunks
            sentence = make_chunk(sentence)
            chunks = {}
            continue
        if line[0] == '*':
            # * 0 -1D 0/0 0.000000 (*これらの数字はstr型)
            # * 文節番号 係り先の文節番号D 主辞/機能語 かかりやすさの度合い
            cols = line.split(' ')
            cols[2] = re.sub(r'(.+?)D', r'\1', cols[2])
            dst = int(cols[2]) #係り先の文節番号
            idx = int(cols[1]) #文節番号
            if idx not in chunks.keys():
                chunks[idx] = Chunk()
                chunks[idx].idx = idx
            chunks[idx].dst = dst
            # dst == -1 => 係り先がない
            if dst != -1:
                if dst not in chunks.keys():
                    chunks[dst] = Chunk()
                    chunks[dst].idx = dst
                chunks[dst].srcs.append(idx)
            continue
        cols = line.split('\t')
        columns = cols[1].split(',')
        morph = Morph(cols[0], columns[6], columns[0], columns[1])
        chunks[idx].morphs.append(morph)
    return sentence


def extract_values(dict):
    dict = sorted(dict.items(), key = lambda x:x[0])
    values = [tuple[1] for tuple in dict]
    return values


def make_chunk(sentence):
    for chunk in sentence:
        chunk.make()
    return sentence

#エッジのリストを返す。
def make_edge_list(sentence):
    edge_list = []
    for i in range(len(sentence)):
        dst = sentence[i].dst
        idx = sentence[i].idx
        if dst == -1:
            #edge_list.append([idx])
            break
        edge_list.append([idx, dst])
    return edge_list


def draw_dependency_tree(sentence):
    sentence = parse_sentence(sentence)
    sentence = make_sentence(sentence)
    edge_list = make_edge_list(sentence)
    #有向グラフを設定
    #左から右へ
    G = pydot.Dot(rankdir = 'LR', graph_type = 'digraph')
    # 文節番号:文節の辞書を作る(e.g.1:"吾輩")
    for chunk in sentence:
        #全ての文節のノードを作る
        node = pydot.Node("%s" % chunk.idx, label = "%s" % chunk.chunk\
            , fontname = font)
        G.add_node(node)

    for edge in edge_list:
        #エッジを作成
        G.add_edge(pydot.Edge("%s" % edge[0], "%s" % edge[1]))

    G.write_png("tree.png")
    print(G.to_string())


def main():
    sentence = input("input sentence :")
    draw_dependency_tree(sentence)

if __name__ == "__main__":
    main()


'''
2016年から2017年にかけて、ディープラーニングを導入したAIが完全情報ゲームである囲碁などの
トップ棋士、さらに不完全情報ゲームであるポーカーの世界トップクラスのプレイヤーも破り、麻雀では
「Microsoft Suphx (Super Phoenix)」がAIとして初めて十段に到達するなど、時代の最先端技術となった。

digraph G {
rankdir=LR;
0 [fontname=IPAGothic, label="2016年から"];
1 [fontname=IPAGothic, label="2017年にかけて"];
2 [fontname=IPAGothic, label="ディープラーニングを"];
3 [fontname=IPAGothic, label="導入した"];
4 [fontname=IPAGothic, label="AIが"];
5 [fontname=IPAGothic, label="完全情報ゲームである"];
6 [fontname=IPAGothic, label="囲碁などの"];
7 [fontname=IPAGothic, label="トップ棋士"];
8 [fontname=IPAGothic, label="さらに"];
9 [fontname=IPAGothic, label="不完全情報ゲームである"];
10 [fontname=IPAGothic, label="ポーカーの"];
11 [fontname=IPAGothic, label="世界トップクラスの"];
12 [fontname=IPAGothic, label="プレイヤーも"];
13 [fontname=IPAGothic, label="破り"];
14 [fontname=IPAGothic, label="麻雀では"];
15 [fontname=IPAGothic, label="MicrosoftSuphx(SuperPhoenix)」が"];
16 [fontname=IPAGothic, label="AIとして"];
17 [fontname=IPAGothic, label="初めて"];
18 [fontname=IPAGothic, label="十段に"];
19 [fontname=IPAGothic, label="到達するなど"];
20 [fontname=IPAGothic, label="時代の"];
21 [fontname=IPAGothic, label="最先端技術と"];
22 [fontname=IPAGothic, label="なった"];
0 -> 1;
1 -> 3;
2 -> 3;
3 -> 4;
4 -> 5;
5 -> 6;
6 -> 7;
7 -> 12;
8 -> 9;
9 -> 10;
10 -> 11;
11 -> 12;
12 -> 13;
13 -> 22;
14 -> 22;
15 -> 19;
16 -> 19;
17 -> 19;
18 -> 19;
19 -> 22;
20 -> 21;
21 -> 22;
}
'''
