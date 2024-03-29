from collections import defaultdict
from knock40 import Morph, morph2sents
from knock41 import Chunk, morph2chunk

if __name__ == "__main__":
    with open("ai.ja1.txt.parsed", "r") as ai:
        ai = ai.readlines()

    ai_morphs = []
    for i in range(len(ai)):
        ai_morphs.append(Morph(ai[i]))

    sent = morph2sents(ai_morphs)[32]  #任意の文章No.を指定
    
    #for sent in sents:
    chunks = morph2chunk(sent)
    sen = []
    f = open("graph.gv", "w", encoding="UTF-8")
    f.write("digraph graph_name {"+ "\n")
    f.write("node [fontname = \"MS Gothic\"];"+ "\n")
    for i in range(len(chunks)):
        goto = chunks[i].dst
        sen.append(chunks[i].show_only_words())
        if goto != None:    
            f.write(chunks[i].show_only_words())
            f.write(" -> "+ chunks[goto].show_only_words())
            goto = chunks[goto].dst
            f.write(";" + "\n")
    f.write("{rank=same; " +", ".join(sen)+";}}")
    f.close()

    #brew install graphviz
    #dot -Tpng graph.gv > dependecy.png
    #今回はGraphviz onlineを使った


