from knock30 import mecab
from collections import defaultdict

if __name__ == "__main__":
    with open("neko.txt.mecab", "r") as neko:
        neko = neko.readlines()

    surfaces = defaultdict(int)

    for i in range(len(neko)):
        nekko = mecab(neko[i])
        if not nekko.EOS:
            surfaces[nekko.surface] += 1
    
    print(sorted(surfaces.items(), key=lambda x:x[1], reverse=True))