from knock40 import Morph, morph2sents
from knock41 import Chunk, morph2chunk, sources

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
        if dep[i].srcs != []:
            print("from", end=" : ")
            print([dep[elem].show_only_words() for elem in dep[i].srcs])
        print(dep[i].show_only_words())
        if dep[i].dst != None:
            print("to : " + dep[dep[i].dst].show_only_words())
        else:
            print("Root")
