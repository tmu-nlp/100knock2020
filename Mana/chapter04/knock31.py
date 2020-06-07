from knock30 import mecab


if __name__ == "__main__":
    with open("neko.txt.mecab", "r") as neko:
        neko = neko.readlines()

    surfaces = set()

    for i in range(len(neko)):
        nekko = mecab(neko[i])
        surfaces.add(nekko.extractpos_surface("動詞"))

    print(surfaces)