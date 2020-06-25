# 41. 係り受け解析結果の読み込み（文節・係り受け）
# 40に加えて，文節を表すクラスChunkを実装せよ．
# このクラスは形態素（Morphオブジェクト）のリスト（morphs），係り先文節インデックス番号（dst），係り元文節インデックス番号のリスト（srcs）をメンバ変数に持つこととする．
# さらに，入力テキストの係り受け解析結果を読み込み，１文をChunkオブジェクトのリストとして表現し，冒頭の説明文の文節の文字列と係り先を表示せよ．
# 本章の残りの問題では，ここで作ったプログラムを活用せよ．

class Morph:
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1
    def __str__(self):
        return f"表層形:{self.surface}\t基本形:{self.base}\t品詞:{self.pos}\t品詞細分類1:{self.pos1}"

class Chunk:
    def __init__(self):
        self.morphs = []
        self.index = -1 # 文節番号。本当は要らない。
        self.dst = -1 # 係り先。-1は係り先がないことを表す。
        self.srcs = [] # 係り元
    def __str__(self):
        surface = ""
        for morph in self.morphs:
            surface += morph.surface
        return f"index:{self.index}\tsurface:{surface}\tdst:{self.dst}\tsrcs{self.srcs}"

def getdata(cabocha_file_path):
    document = []
    # morphs、index、dstを埋める
    with open(cabocha_file_path, "r") as cabocha_file:
        sentence = [] # chunkのリストになる
        chunk = Chunk()
        for line in cabocha_file:
            line = line.strip()
            if line == "EOS":
                if len(chunk.morphs) > 0:
                    sentence.append(chunk)
                document.append(sentence)
                sentence = []
            elif line[0] == "*":
                line = line.split()
                if line[1] != "0":
                    sentence.append(chunk)
                chunk = Chunk()
                chunk.index = int(line[1])
                line[2] = line[2].replace("D", "")
                line[2] = int(line[2])
                chunk.dst = line[2]
            else:
                line = line.split("\t")
                col1 = line[0]
                col2 = line[1].split(",")
                morph = Morph(col1, col2[6], col2[0], col2[1])
                chunk.morphs.append(morph)
    # srcsを埋める
    for sentence in document:
        for chunk in sentence:
            dst = chunk.dst
            if dst == -1:
                continue
            sentence[dst].srcs.append(chunk.index)
    return document

if __name__ == "__main__":
    cabocha_file_path = "ai.ja.split.txt.parsed"
    document = getdata(cabocha_file_path)
    for i, sentence in enumerate(document[:4]):
        print(f"-----{i+1}文目-----")
        for chunk in sentence:
            print(chunk)