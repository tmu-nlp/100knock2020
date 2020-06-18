"""
46. 動詞の格フレーム情報の抽出Permalink
45のプログラムを改変し，述語と格パターンに続けて項（述語に係っている文節そのもの）をタブ区切り形式で出力せよ．
45の仕様に加えて，以下の仕様を満たすようにせよ．

・項は述語に係っている文節の単語列とする（末尾の助詞を取り除く必要はない）
・述語に係る文節が複数あるときは，助詞と同一の基準・順序でスペース区切りで並べる
「ジョン・マッカーシーはAIに関する最初の会議で人工知能という用語を作り出した。」という例文を考える． 
この文は「作り出す」という１つの動詞を含み，「作り出す」に係る文節は
「ジョン・マッカーシーは」，「会議で」，「用語を」であると解析された場合は，次のような出力になるはずである．

[作り出す	で は を	会議で ジョンマッカーシーは 用語を]
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
                #動詞があったとき
                for src in chunk.srcs:
                    #その文節に掛かる文節の助詞を抽出
                    post_list = chunks[src].get_morphs_by_pos("助詞")
                    #助詞があるときに文節内の最後の助詞を選ぶ&その文節を抽出
                    if len(post_list) > 0:
                        posts.append(post_list[-1])
                        chu = "".join([sur.surface for sur in chunks[src].morphs])
                        chus.append(chu)
            if len(posts) < 1:
                continue
            #助詞のリストと助詞を含んだ文節のリストを同時にソートする
            mixed = zip([post.surface for post in posts], chus)
            mixed = sorted(mixed)
            posts, chus = zip(*mixed)

            print(("{}\t{}\t{}".format(verbs[0].base, ' '.join(posts), ' '.join(chus))))
