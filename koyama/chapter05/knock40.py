# 40. 係り受け解析結果の読み込み（形態素）
# 形態素を表すクラスMorphを実装せよ．
# このクラスは表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をメンバ変数に持つこととする．
# さらに，係り受け解析の結果（ai.ja.txt.parsed）を読み込み，各文をMorphオブジェクトのリストとして表現し，冒頭の説明文の形態素列を表示せよ．

class Morph:
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1
    def __str__(self):
        return f"表層形:{self.surface}\t基本形:{self.base}\t品詞:{self.pos}\t品詞細分類1:{self.pos1}"

def getdata１(cabocha_file_path):
    document = []
    with open(cabocha_file_path, "r") as cabocha_file:
        sentence = []
        for line in cabocha_file:
            line = line.strip()
            if line == "EOS":
                document.append(sentence)
                sentence = []
            elif line[0] == "*":
                continue
            else:
                line = line.split("\t")
                col1 = line[0]
                col2 = line[1].split(",")
                morph = Morph(col1, col2[6], col2[0], col2[1])
                sentence.append(morph)
    return document

if __name__ == "__main__":
    cabocha_file_path = "ai.ja.split.txt.parsed"
    document = getdata１(cabocha_file_path)
    for i, sentence in enumerate(document[:5]):
        print(f"-----{i+1}文目-----")
        for morph in sentence:
            print(morph)

# CaboChaでファイルごと解析するには以下のコマンドを実行する
# cabocha -f1 -o ai.ja.txt.parsed ai.ja.txt