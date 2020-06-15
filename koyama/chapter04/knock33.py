from knock30 import getdata

if __name__ == "__main__":
    mecab_file_path = "neko.txt.mecab"
    document = getdata(mecab_file_path)
    a_no_b = []
    for sentence in document:
        for i in range(1, len(sentence) - 1):
            if sentence[i - 1]["pos"] == "名詞" and sentence[i]["surface"] == "の" and sentence[i + 1]["pos"] == "名詞":
                a_no_b.append(sentence[i - 1]['surface'] + sentence[i]['surface'] + sentence[i + 1]['surface'])
    print(a_no_b)