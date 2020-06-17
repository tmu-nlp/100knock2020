from knock40 import Morph
from knock40 import morph2sents

class Chunk():
    def __init__(self, morphlist):
        self.morphs = morphlist[1:]
        #self.morphs = [elem.show() for elem in morphlist[1:]]見やすい
        self.meta = morphlist[0].show()
        if self.meta[2] != "-1D":
            self.dst = int(self.meta[2][:-1])
        else:
            self.dst = None
        self.srcs = []

    def show_bunsetsu_tag(self):
        return "".join([elem.show() for elem in self.morphs])

    def show_morph_pos1(self):
        return [elem.pos1 for elem in self.morphs]

    def show_only_words(self):
        words = []
        for elem in self.morphs:
            if elem.pos != "記号":
                words.append(elem.surface)
        return "".join(words)
    
    def show_only_listwords(self):
        words = []
        for elem in self.morphs:
            if elem.pos != "記号":
                words.append(elem.surface)
        return words

    def show_only_listpos(self):
        words = []
        for elem in self.morphs:
            words.append(elem.pos)
        return words    

    def show_base_for_X(self, X):
        for elem in self.morphs:
            if elem.pos == X:
                return elem.base
    
    def replace_for_X(self, A):
        words = []
        for elem in self.morphs:
            if elem.pos != "記号":
                if elem.pos != "名詞":
                    words.append(elem.surface)
                else:
                    words.append(A)
        return "".join(words)
    
def morph2chunk(morphlists):
    morphlists.append(Morph("*"))
    chunks = []
    chunk = []
    for elem in morphlists:
        #print(elem.show())
        if elem.show()[0] != "*" :
            chunk.append(elem)
        else:
            if chunk != []:
                chunks.append(Chunk(chunk))
            chunk = []
            chunk.append(elem)
    return chunks

def sources(chunklist):
    for i in range(len(chunklist)):
        if chunklist[i].dst != None:
            chunklist[chunklist[i].dst].srcs.append(i)

if __name__ == "__main__":
    with open("ai.ja.txt.parsed", "r") as ai:
        ai = ai.readlines()

    ai_morphs = []
    for i in range(len(ai)):
        ai_morphs.append(Morph(ai[i]))
    
    #print(morph2sent(ai_morphs)[1][0].show())
    dep = morph2chunk(morph2sents(ai_morphs)[1])
    #print(dep)
    sources(dep)

    for i in range(len(dep)):
        print(dep[i].show_bunsetsu_tag(), end="\t")
        if dep[i].dst != None:
            print(dep[dep[i].dst].show_bunsetsu_tag())
        else:
            print("Root")

"""
人工知能        語。
（じんこうちのう、、    語。
AI      〈エーアイ〉）とは、
〈エーアイ〉）とは、    語。
「『計算        （）』という
（）』という    道具を
概念と  道具を
『コンピュータ  （）』という
（）』という    道具を
道具を  用いて
用いて  研究する
『知能』を      研究する
研究する        計算機科学
計算機科学      （）の
（）の  一分野」を
一分野」を      指す
指す    語。
語。    研究分野」とも
「言語の        推論、
理解や  推論、
推論、  問題解決などの
問題解決などの  知的行動を
知的行動を      代わって
人間に  代わって
代わって        行わせる
コンピューターに        行わせる
行わせる        技術」、または、
技術」、または、        研究分野」とも
「計算機        （コンピュータ）による
（コンピュータ）による  情報処理システムの
知的な  情報処理システムの
情報処理システムの      実現に関する
設計や  実現に関する
実現に関する    研究分野」とも
研究分野」とも  される。
される。        される。
"""