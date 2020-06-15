def getdata(mecab_file_path):
    document = []
    with open(mecab_file_path, "r") as mecab_file:
        sentence = []
        for line in mecab_file:
            if line[0] == "　":
                word = {'surface': "　", 'base': "　", 'pos': "記号", 'pos1': "空白"}
                sentence.append(word)
                continue
            col1 = line.strip().split("\t")
            if col1[0] == "EOS":
                if len(sentence) == 0:
                    continue
                document.append(sentence)
                sentence = []
                continue
            col2 = col1[1].split(",")
            word = {'surface': col1[0], 'base': col2[6], 'pos': col2[0], 'pos1': col2[1]}
            sentence.append(word)
    return document

if __name__ == "__main__":
    mecab_file_path = "neko.txt.mecab"
    document = getdata(mecab_file_path)
    print(document)

# neko.txtをMeCabを使って形態素解析して，neko.txt.mecabに保存したい
# 以下のコマンドを実行したらできる
# mecab neko.txt -o neko.txt.mecab
