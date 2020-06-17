from knock40 import Morph, morph2sents
from knock41 import Chunk, morph2chunk

def noun2verb(chunks):
    noun = set()
    verb = set()
    for i in range(len(chunks)):
        for morph in chunks[i].morphs:
            noun.add(morph.pos)
            if "名詞" in noun:
                goto = chunks[i].dst
                for morph2 in chunks[goto].morphs:
                    verb.add(morph2.pos)
        if "動詞" in verb:
            print(chunks[i].show_only_words(), end="\t")
            print(chunks[goto].show_only_words())
        noun = set()
        verb = set()

if __name__ == "__main__":
    with open("ai.ja1.txt.parsed", "r") as ai:
        ai = ai.readlines()

    ai_morphs = []
    for i in range(len(ai)):
        ai_morphs.append(Morph(ai[i]))

    dep = morph2chunk(morph2sents(ai_morphs)[1])

    noun2verb(dep)


