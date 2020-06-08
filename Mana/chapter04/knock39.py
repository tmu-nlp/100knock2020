from knock30 import mecab
from collections import defaultdict
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open("neko.txt.mecab", "r") as neko:
        neko = neko.readlines()

    surfaces = defaultdict(int)

    for i in range(len(neko)):
        nekko = mecab(neko[i])
        if not nekko.EOS:
            surfaces[nekko.surface] += 1
    
    res = sorted(surfaces.items(), key=lambda x:x[1], reverse=True)

    count = defaultdict(int)
    
    for elem in res:
        count[elem[1]] += 1

    res = sorted(count.items())
    #for elem in res:
        #print(elem[0])
        #print(res[elem])
    
    fig = plt.figure()
    left = [elem[0] for elem in res]
    #print(left)
    height = [elem[1] for elem in res]
    #print(height)
    plt.scatter(left, height)
    plt.yscale("log")
    plt.xscale("log")
    fig.savefig("imagezipf.png")
    