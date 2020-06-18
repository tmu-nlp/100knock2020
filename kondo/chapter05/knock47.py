"""
47. 機能動詞構文のマイニングPermalink
動詞のヲ格にサ変接続名詞が入っている場合のみに着目したい．46のプログラムを以下の仕様を満たすように改変せよ．

・「サ変接続名詞+を（助詞）」で構成される文節が動詞に係る場合のみを対象とする
・述語は「サ変接続名詞+を+動詞の基本形」とし，文節中に複数の動詞があるときは，最左の動詞を用いる
・述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる
・述語に係る文節が複数ある場合は，すべての項をスペース区切りで並べる（助詞の並び順と揃えよ）
例えば「また、自らの経験を元に学習を行う強化学習という手法もある。」という文から，以下の出力が得られるはずである．
[学習を行う	に を	元に 経験を]
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
    for chunks in chunk_to_list():
        for chunk in chunks:
            posts = []
            chus = []
            #文節内の動詞を抽出
            verbs = chunk.get_morphs_by_pos("動詞")
            if len(verbs) > 0:
                #動詞　があったとき
                for src in chunk.srcs:
                    #その文節に掛かる文節の助詞を抽出
                    post_list = chunks[src].get_morphs_by_pos("助詞")
                    #その文節に掛かる文節の　サ変を　を抽出
                    sa_wo = chunks[src].get_sa_wo()
                    #助詞があるときに文節内の最後の助詞を選ぶ&その文節を抽出
                    if len(post_list) > 0:
                        posts.append(post_list[-1])
                        chu = "".join([sur.surface for sur in chunks[src].morphs])
                        chus.append(chu)
            if len(posts) < 1 or len(sa_wo) < 1:
                continue
            mixed = zip([post.surface for post in posts], chus)
            mixed = sorted(mixed)
            posts, chus = zip(*mixed)

            print(("{}{}\t{}\t{}".format(sa_wo, verbs[0].base, ' '.join(posts), ' '.join(chus))))