'''
40. 係り受け解析結果の読み込み（形態素）Permalink
形態素を表すクラスMorphを実装せよ．このクラスは表層形（surface），基本形（base），品詞（pos），
品詞細分類1（pos1）をメンバ変数に持つこととする．さらに，係り受け解析の結果（ai.ja.txt.parsed）
を読み込み，各文をMorphオブジェクトのリストとして表現し，冒頭の説明文の形態素列を表示せよ．
'''
#表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音
import re
import io
fname = "ai.ja.txt.parsed"

class Morph:
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1

    def print(self):
        print("[surface] : {}\t[base] : {}\t[pos] : {}\t[pos1] : {}".format(\
            self.surface, self.base, self.pos, self.pos1))

def text_processing(file):
    with open(file) as f:
        text = f.read()
        # *から始まる行（文節の係り先とかの説明）を消す
        c = re.compile(r'^\*\s\d.+?\n', re.MULTILINE)
        text_processed = c.sub('', text)
        # 段落のための改行（EOSが二つ連続で出現しているところ）を消す
        c = re.compile(r'(EOS\n)+')
        text_processed = c.sub(r'EOS\n', text_processed)
    return text_processed

def sentence_list(text):
    file = io.StringIO(text)
    sentences = []
    #一文中の形態素を集めたリスト
    morphs = []
    for line in file:
        if line == 'EOS\n':
            sentences.append(morphs)
            morphs = []
            continue
        cols = line.split('\t')
        columns = cols[1].split(',')
        morph = Morph(cols[0], columns[6], columns[0], columns[1])
        morphs.append(morph)

    return sentences

def main():
    text = text_processing(fname)
    sentences = sentence_list(text)
    # 冒頭の説明は1、2番目の文
    cnt = 0
    for sentence in sentences:
        if cnt == 0:
            cnt += 1
            continue
        if cnt == 3: break
        for morph in sentence:
            morph.print()
        cnt += 1

if __name__ == "__main__":
    main()
