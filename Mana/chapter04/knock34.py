from knock30 import mecab, morph2sent

if __name__ == "__main__":
    with open("neko.txt.mecab", "r") as neko:
        neko = neko.readlines()
        
    nekos = []
    for i in range(len(neko)):
        nekos.append(mecab(neko[i]))

    sent_org = morph2sent(nekos)

    noun = []
    noun_seq = []
    seq = True
   
    for elem in sent_org.values():
        for i in range(len(elem)):
            if seq and elem[i][1].pos == "名詞":
                noun.append(elem[i][0])
            elif seq and elem[i][1].pos != "名詞":
                if len(noun) >1:
                    noun_seq.append("".join(noun))
                seq = False
            elif not seq and elem[i][1].pos == "名詞":
                noun = []
                noun.append(elem[i][0])
                seq = True

    print(noun_seq)
