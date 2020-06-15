from knock40 import Morph, morph2sent
from knock41 import Chunk, morph2chunk

if __name__ == "__main__":
    with open("ai.ja.txt.parsed", "r") as ai:
        ai = ai.readlines()

    ai_morphs = []
    for i in range(len(ai)):
        ai_morphs.append(Morph(ai[i]))

    dep = morph2chunk(morph2sent(ai_morphs)[2])

    for elem in dep:
        print(elem.srcs)
        print(elem.dst)
        print(elem.show_bunsetsu_tag())