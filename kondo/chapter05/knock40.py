"""
40. 係り受け解析結果の読み込み（形態素）Permalink
形態素を表すクラスMorphを実装せよ．
このクラスは表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をメンバ変数に持つこととする．
さらに，係り受け解析の結果（ai.ja.txt.parsed）を読み込み，各文をMorphオブジェクトのリストとして表現し，
冒頭の説明文の形態素列を表示せよ．
"""

data = "ai.ja.txt.parsed"

class Morph:
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1

    def __str__(self):
        return "surface[{}]\tbase[{}]\tpos[{}]\tpos1[{}]".format(self.surface, self.base, self.pos, self.pos1)

def morph_to_list():
    with open(data, encoding = 'utf-8') as cabocha:
        res = []
        for line in cabocha:
            if line == "EOS\n":
                if len(res) > 0:
                    yield res
                    res = []
            else:
                if line[0] == '*':
                    continue
                surface, els = line.split('\t')
                els = els.split(',')

                res.append(Morph(surface, els[6], els[0], els[1]))

        #raise StopIterationはRunTimeEror出る(python3.7以降)
        return

if __name__ == "__main__":
    for i, morphs in enumerate(morph_to_list()):
        for morph in morphs:
            print(morph)
        print("\n")
        if i == 2:
            break