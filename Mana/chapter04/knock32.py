from knock30 import mecab

if __name__ == "__main__":
    with open("neko.txt.mecab", "r") as neko:
        neko = neko.readlines()

    bases = set()

    for i in range(len(neko)):
        nekko = mecab(neko[i])
        bases.add(nekko.extractpos_base("動詞"))

    print(bases)