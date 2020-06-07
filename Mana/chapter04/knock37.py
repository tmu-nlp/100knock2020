from knock30 import mecab, morph2sent
import matplotlib.pyplot as plt
from collections import defaultdict

if __name__ == "__main__":
    with open("neko.txt.mecab", "r") as neko:
        neko = neko.readlines()

    nekos = []
    for i in range(len(neko)):
        nekos.append(mecab(neko[i]))

    sent_org = morph2sent(nekos)

    count = defaultdict(int)

    #一文に猫が二回でたら重複してる？要チェック

    for key, value in sent_org.items():
        if "猫" in key:
            #print(elem)
            for word in value:
                    count[word[0]] += 1

    res = sorted(count.items(), key=lambda x:x[1], reverse=True)[:10]
    #print(res)

    #フォントの問題
    fig = plt.figure()
    left = [elem[0] for elem in res]
    print(left)
    height = [elem[1] for elem in res]
    print(height)
    plt.plot(left, height)
    fig.savefig("imageNeko.png")
