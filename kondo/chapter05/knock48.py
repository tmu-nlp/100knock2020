"""
48. 名詞から根へのパスの抽出Permalink
文中のすべての名詞を含む文節に対し，その文節から構文木の根に至るパスを抽出せよ． ただし，構文木上のパスは以下の仕様を満たすものとする．

各文節は（表層形の）形態素列で表現する
パスの開始文節から終了文節に至るまで，各文節の表現を” -> “で連結する
「ジョン・マッカーシーはAIに関する最初の会議で人工知能という用語を作り出した。」という例文を考える． 
CaboChaを係り受け解析に用いた場合，次のような出力が得られると思われる．

[
ジョンマッカーシーは -> 作り出した
AIに関する -> 最初の -> 会議で -> 作り出した
最初の -> 会議で -> 作り出した
会議で -> 作り出した
人工知能という -> 用語を -> 作り出した
用語を -> 作り出した
]

KNPを係り受け解析に用いた場合，次のような出力が得られると思われる．

[
ジョンマッカーシーは -> 作り出した
ＡＩに -> 関する -> 会議で -> 作り出した
会議で -> 作り出した
人工知能と -> いう -> 用語を -> 作り出した
用語を -> 作り出した
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
            for chunk in chunks:
                posts = []
                chus = []
                #文節内の動詞を抽出
                nouns = chunk.get_morphs_by_pos("名詞")
                if len(nouns) > 0:
                    #名詞　があったとき
                    next = chunk
                    while True:
                        #記号いらないならnext.normalized_surface()
                        print(next.print_chunk(), end = "")
                        next = chunks[next.dst]
                        if next.dst == -1:
                            print("\n")
                            break
                        else:
                            print(" -> ", end = "")