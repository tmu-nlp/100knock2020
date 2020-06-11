from knock30 import getdata

if __name__ == "__main__":
    mecab_file_path = "neko.txt.mecab"
    document = getdata(mecab_file_path)
    lns = []
    for sentence in document:
        nouns = []
        tmp = []
        for i in range(len(sentence)):
            if sentence[i]["pos"] == "名詞":
                tmp.append(sentence[i]["surface"])
            else:
                if len(tmp) >= 2:
                    nouns.append(tmp)
                tmp = []
        for noun in nouns:
            lns.append("".join(noun))
    print(lns)





