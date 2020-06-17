from collections import defaultdict
from knock40 import Morph, morph2sents
from knock41 import Chunk, morph2chunk, sources


if __name__ == "__main__":
    with open("ai.ja.txt.parsed", "r") as ai:
        ai = ai.readlines()

    ai_morphs = []
    for i in range(len(ai)):
        ai_morphs.append(Morph(ai[i]))

    sents = morph2sents(ai_morphs)
    for sent in sents:
        chunks = morph2chunk(sent)
        sources(chunks)
        for i in range(len(chunks)):
            #print(chunk.meta)
            #print(chunk.show_bunsetsu_tag())
            if "サ変接続" in chunks[i].show_morph_pos1():
                if "を" in chunks[i].show_only_listwords():
                    print(chunks[i].show_only_words(), end="")
                    goto = chunks[i].dst
                    verb = chunks[goto].show_base_for_X("動詞")
                    adpos = set()
                    dep = []
                    if verb != None:
                        if chunks[goto].srcs != []:
                            for head_id in chunks[goto].srcs:
                                adp = chunks[head_id].show_base_for_X("助詞")
                                if adp != None and head_id != i:
                                    adpos.add(adp)
                                    dep.append(chunks[head_id].show_only_words())
                            print(verb, end="\t")
                            print(" ".join(adpos), end="\t")
                            print(" ".join(dep))
                        else:
                            print(verb)

