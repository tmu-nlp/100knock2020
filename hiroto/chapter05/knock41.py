'''
41. 係り受け解析結果の読み込み（文節・係り受け）Permalink
40に加えて，文節を表すクラスChunkを実装せよ．このクラスは形態素（Morphオブジェクト）の
リスト（morphs），係り先文節インデックス番号（dst），係り元文節インデックス番号のリスト（srcs）
をメンバ変数に持つこととする．さらに，入力テキストの係り受け解析結果を読み込み，１文を
Chunkオブジェクトのリストとして表現し，冒頭の説明文の文節の文字列と係り先を表示せよ．
本章の残りの問題では，ここで作ったプログラムを活用せよ．
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

class Chunk:
    def __init__(self):
        self.morphs = []
        self.dst = -1
        self.srcs = []
        self.idx = -1
    #Chunkオブジェクトの形態素列を文字列にまとめる
    def make(self):
        chunk = ''
        for morph in self.morphs:
            chunk += morph.surface
        #文節のテキストを表す
        self.chunk = chunk

    def print(self):
        print("[文節番号:{}] {}\t[係り先:{}]\t[係り元{}]".format(self.idx, self.chunk\
                ,self.dst, self.srcs))


def text_processing(file):
    with open(file) as f:
        text = f.read()
        # 段落のための改行（EOSが二つ連続で出現しているところ）を消す
        c = re.compile(r'EOS\nEOS\n')
        text_processed = c.sub(r'EOS\n', text)
    return text_processed

#一文中の文節を集めた辞書のkey（文節番号）順にソートし、そのChunkオブジェクトのリストを返す
def extract_values(dic):
    dic = sorted(dic.items(), key = lambda x:x[0])
    values = [tuple[1] for tuple in dic]
    return values

def sentence_list(text):
    file = io.StringIO(text)
    idx = 0
    sentences = []
    #一文中の文節を集めた辞書 (key:文節番号, value:Chunkオブジェクト)
    chunks = {}
    for line in file:
        if line == 'EOS\n':
            chunks = extract_values(chunks)
            sentences.append(chunks)
            chunks = {}
            continue
        if line[0] == '*':
            # * 0 -1D 0/0 0.000000 (*これらの数字はstr型)
            # * 文節番号 係り先の文節番号D 主辞/機能語 かかりやすさの度合い
            cols = line.split(' ')
            #Dを消して、係り先の文節番号だけ抜き取る
            cols[2] = re.sub(r'(.+?)D', r'\1', cols[2])
            dst = int(cols[2]) #係り先の文節番号
            idx = int(cols[1]) #文節番号
            if idx not in chunks.keys():
                #Chunkインスタンスを作成
                chunks[idx] = Chunk()
                chunks[idx].idx = idx
            chunks[idx].dst = dst
            # dst == -1 => 係り先がない
            if dst != -1:
                if dst not in chunks.keys():
                    chunks[dst] = Chunk()
                    chunks[dst].idx = dst
                chunks[dst].srcs.append(idx)
            continue
        cols = line.split('\t')
        columns = cols[1].split(',')
        morph = Morph(cols[0], columns[6], columns[0], columns[1])
        chunks[idx].morphs.append(morph)
    return sentences

def main():
    text = text_processing(fname)
    sentences = sentence_list(text)

    cnt = 0
    for sentence in sentences:
        if cnt == 0:
            cnt += 1
            continue
        if cnt == 3: break
        for chunk in sentence:
            chunk.make()
            chunk.print()
        cnt += 1


if __name__ == "__main__":
    main()
