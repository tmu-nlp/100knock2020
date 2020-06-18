"""
49. 名詞間の係り受けパスの抽出Permalink
文中のすべての名詞句のペアを結ぶ最短係り受けパスを抽出せよ．
ただし，名詞句ペアの文節番号がiとj（i<j）のとき，係り受けパスは以下の仕様を満たすものとする．

・問題48と同様に，パスは開始文節から終了文節に至るまでの各文節の表現（表層形の形態素列）を” -> “で連結して表現する
・文節iとjに含まれる名詞句はそれぞれ，XとYに置換する
また，係り受けパスの形状は，以下の2通りが考えられる．

・文節iから構文木の根に至る経路上に文節jが存在する場合: 文節iから文節jのパスを表示
・上記以外で，文節iと文節jから構文木の根に至る経路上で共通の文節kで交わる場合: 文節iから文節kに至る直前のパスと文節jから文節kに至る直前までのパス，文節kの内容を” | “で連結して表示
「ジョン・マッカーシーはAIに関する最初の会議で人工知能という用語を作り出した。」という例文を考える． 
CaboChaを係り受け解析に用いた場合，次のような出力が得られると思われる．

[
Xは | Yに関する -> 最初の -> 会議で | 作り出した
Xは | Yの -> 会議で | 作り出した
Xは | Yで | 作り出した
Xは | Yという -> 用語を | 作り出した
Xは | Yを | 作り出した
Xに関する -> Yの
Xに関する -> 最初の -> Yで
Xに関する -> 最初の -> 会議で | Yという -> 用語を | 作り出した
Xに関する -> 最初の -> 会議で | Yを | 作り出した
Xの -> Yで
Xの -> 会議で | Yという -> 用語を | 作り出した
Xの -> 会議で | Yを | 作り出した
Xで | Yという -> 用語を | 作り出した
Xで | Yを | 作り出した
Xという -> Yを
]

KNPを係り受け解析に用いた場合，次のような出力が得られると思われる．

[
Xは | Yに -> 関する -> 会議で | 作り出した。
Xは | Yで | 作り出した。
Xは | Yと -> いう -> 用語を | 作り出した。
Xは | Yを | 作り出した。
Xに -> 関する -> Yで
Xに -> 関する -> 会議で | Yと -> いう -> 用語を | 作り出した。
Xに -> 関する -> 会議で | Yを | 作り出した。
Xで | Yと -> いう -> 用語を | 作り出した。
Xで | Yを | 作り出した。
Xと -> いう -> Yを
]
"""

data = "ai.ja.txt.parsed"
ans_file = "ans_45.txt"

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

    def get_morphs_by_pos(self, pos):
        #こういうlist返すときシンプルにかける
        return [res for res in self.morphs if res.pos == pos]
    
    def get_morphs_by_pos1(self, pos1):
        #こういうlist返すときシンプルにかける
        return [res for res in self.morphs if res.pos1 == pos1]
    
    def get_sa_wo(self):
        for i, morph in enumerate(self.morphs[0:-1]):
            if morph.pos1 == "サ変接続" and self.morphs[i + 1].surface == 'を':
                return morph.surface + self.morphs[i + 1].surface
        return ""
    
    def print_chunk(self):
        word = ""
        for morph in self.morphs:
            word += morph.surface
        return word
    
    def noun_trans(self, str):
        re = ""
        for morph in self.morphs:
            if morph.pos != "記号":
                if morph.pos == "名詞":
                    re += str
                    str = ""
                else: re += morph.surface
        return re

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
                if num not in chunks:
                    chunks[num] = Chunk()
                chunks[num].dst = dst
                if dst != -1:
                    if dst not in chunks:
                        chunks[dst] = Chunk()
                    chunks[dst].srcs.append(num)
            else:
                surface, els = line.split('\t')
                els = els.split(',')

                chunks[num].morphs.append(Morph(surface, els[6], els[0], els[1]))

        #raise StopIterationはRunTimeEror出る(python3.7以降)
        return

if __name__ == "__main__":
    for i, chunks in enumerate(chunk_to_list()):
        if i == 13:
            including_noun = [i for i in range(len(chunks)) if len(chunks[i].get_morphs_by_pos("名詞"))]
            if len(including_noun) < 2:
                continue
            for i, x in enumerate(including_noun[:-1]):
                for y in including_noun[i + 1:]:
                    collision = 0   #xとyが衝突したかどうか
                    route_col = -1  #経路の衝突場所
                    routes_x = set()    #xの辿ったルート
                    #xから根まで探索　yにぶつかったら終了
                    dst = chunks[x].dst
                    while dst != -1:
                        if dst == y:    #xとyの衝突
                            collision = 1
                            break
                        #xのルートを保持しておく
                        routes_x.add(dst)
                        dst = chunks[dst].dst

                    #yから根まで探索　xの経路にぶつからないか探索
                    if collision == 0:  #xとyの衝突がないとき
                        dst = chunks[y].dst
                        while dst != -1:
                            if dst in routes_x: #経路の衝突
                                route_col = dst
                                break
                            else:
                                dst = chunks[dst].dst
                    
                    if route_col == -1:
                        print(chunks[x].noun_trans('X'), end = "")
                        dst = chunks[x].dst
                        while dst != -1:
                            if dst == y:
                                print(" -> " + chunks[dst].noun_trans('Y'), end = "")
                                break
                            else:
                                print(" -> " + chunks[dst].normalized_surface(), end = "")
                            dst = chunks[dst].dst
                        print("")

                    else:
                        #xから衝突手前まで
                        print(chunks[x].noun_trans('X'), end = "")
                        dst = chunks[x].dst
                        while dst != route_col:
                            print(" -> " + chunks[dst].normalized_surface(), end = "")
                            dst = chunks[dst].dst
                        print(" | ", end = "")

                        #yから衝突手前まで
                        print(chunks[y].noun_trans('Y'), end = "")
                        dst = chunks[y].dst
                        while dst != route_col:
                            print(" -> " + chunks[dst].normalized_surface(), end = "")
                            dst = chunks[dst].dst
                        print(" | ", end = "")
                        #衝突したところ
                        print(chunks[route_col].normalized_surface(), end = "")
                        print("")
