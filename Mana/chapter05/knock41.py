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
        words = set()
        for elem in self.morphs:
            if elem.pos != "記号":
                if elem.pos == "名詞":
                    words.add(A)
                else:
                    words.add(elem.surface)
        return "".join(sorted(words))
    
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
    with open("ai.ja1.txt.parsed", "r") as ai:
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