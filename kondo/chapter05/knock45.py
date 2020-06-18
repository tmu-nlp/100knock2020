"""
45. 動詞の格パターンの抽出Permalink
今回用いている文章をコーパスと見なし，日本語の述語が取りうる格を調査したい． 
動詞を述語，動詞に係っている文節の助詞を格と考え，述語と格をタブ区切り形式で出力せよ． 
ただし，出力は以下の仕様を満たすようにせよ．

・動詞を含む文節において，最左の動詞の基本形を述語とする
・述語に係る助詞を格とする
・述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる

「ジョン・マッカーシーはAIに関する最初の会議で人工知能という用語を作り出した。」という例文を考える． 
この文は「作り出す」という１つの動詞を含み，「作り出す」に係る文節は「ジョン・マッカーシーは」，「会議で」，「用語を」であると解析された場合は，
次のような出力になるはずである．

[作り出す	で は を]
このプログラムの出力をファイルに保存し，以下の事項をUNIXコマンドを用いて確認せよ．

・コーパス中で頻出する述語と格パターンの組み合わせ
・「行う」「なる」「与える」という動詞の格パターン（コーパス中で出現頻度の高い順に並べよ）
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
    with open(ans_file, "w", encoding="utf-8") as file_45:
        for chunks in chunk_to_list():
            for chunk in chunks:
                posts = []
                #文節内の動詞を抽出
                verbs = chunk.get_morphs_by_pos("動詞")
                if len(verbs) > 0:
                    #動詞があったとき
                    for src in chunk.srcs:
                        #その文節に掛かる文節の助詞を抽出
                        post_list = chunks[src].get_morphs_by_pos("助詞")
                        #文節内の最後の助詞を選ぶ
                        if len(post_list) > 0:
                            posts.append(post_list[-1])
                if len(posts) < 1:
                    continue
                
                print("{}\t{}".format(verbs[0].base, ' '.join(sorted(post.surface for post in posts))))
                file_45.write(("{}\t{}\n".format(verbs[0].base, ' '.join(sorted(post.surface for post in posts)))))

#cat ans_45.txt | sort | uniq -c | sort -n --reverse
#cat ans_45.txt | grep "^行う\s" | sort | uniq -c | sort -n --reverse
#cat ans_45.txt | grep "^なる\s" | sort | uniq -c | sort -n --reverse
#cat ans_45.txt | grep "^与える\s" | sort | uniq -c | sort -n --reverse