from knock30 import mecab, morph2sent

if __name__ == "__main__":
    with open("neko.txt.mecab", "r") as neko:
        neko = neko.readlines()
    
    nekos = []
    for i in range(len(neko)):
        nekos.append(mecab(neko[i]))

    sent_org = morph2sent(nekos)

    for elem in sent_org.values():    
        for i in range(1, len(elem)-1):
            if elem[i][0] == "の" and elem[i-1][1].pos== "名詞" and elem[i+1][1].pos== "名詞":
                print("".join([word[0] for word in elem[i-1:i+2]]))

