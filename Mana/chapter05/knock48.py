from collections import defaultdict
from knock40 import Morph, morph2sents
from knock41 import Chunk, morph2chunk

if __name__ == "__main__":
    with open("ai.ja.txt.parsed", "r") as ai:
        ai = ai.readlines()

    ai_morphs = []
    for i in range(len(ai)):
        ai_morphs.append(Morph(ai[i]))

    sents = morph2sents(ai_morphs)
    
    for sent in sents:
        chunks = morph2chunk(sent)
        for i in range(len(chunks)):
            if "名詞" in chunks[i].show_only_listpos():
                #print(chunks[i].show_only_listpos())
                print(chunks[i].show_only_words(), end="")
                goto = chunks[i].dst
                while goto != None:
                    print(" -> "+ chunks[goto].show_only_words(), end="")
                    goto = chunks[goto].dst
                print("")