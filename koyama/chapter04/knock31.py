from knock30 import getdata

if __name__ == "__main__":
    mecab_file_path = "neko.txt.mecab"
    document = getdata(mecab_file_path)
    verb_surface = []
    for sentence in document:
        for word in sentence:
            if word["pos"] == "動詞":
                verb_surface.append(word["surface"])
    print(verb_surface)
