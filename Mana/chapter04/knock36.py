from knock30 import mecab
from collections import defaultdict
import matplotlib.pyplot as plt
import japanize_matplotlib

if __name__ == "__main__":
    with open("neko.txt.mecab", "r") as neko:
        neko = neko.readlines()

    surfaces = defaultdict(int)

    for i in range(len(neko)):
        nekko = mecab(neko[i])
        if not nekko.EOS:
            surfaces[nekko.surface] += 1
    
    res = sorted(surfaces.items(), key=lambda x:x[1], reverse=True)[:10]
    #for elem in res:
        #print(elem[0])
        #print(res[elem])

    #matplotlib日本語化 --> pip install japanize-matplotlib
    #c.f. https://qiita.com/uehara1414/items/6286590d2e1ffbf68f6c
    
    fig = plt.figure()
    left = [elem[0] for elem in res]
    height = [elem[1] for elem in res]
    plt.bar(left, height)
    fig.savefig("image.png")