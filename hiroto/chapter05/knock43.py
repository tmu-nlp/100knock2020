'''
43. 名詞を含む文節が動詞を含む文節に係るものを抽出Permalink
名詞を含む文節が，動詞を含む文節に係るとき，これらをタブ区切り形式で抽出せよ．
ただし，句読点などの記号は出力しないようにせよ．
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

    def make(self):
        chunk = ''
        for morph in self.morphs:
            # 記号を除去
            if morph.pos == "記号": continue
            chunk += morph.surface
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

def sentence_list(text):
    file = io.StringIO(text)
    sentences = []
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
            cols[2] = re.sub(r'(.+?)D', r'\1', cols[2])
            dst = int(cols[2]) #係り先の文節番号
            idx = int(cols[1]) #文節番号
            if idx not in chunks.keys():
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

def extract_values(dict):
    dict = sorted(dict.items(), key = lambda x:x[0])
    values = [tuple[1] for tuple in dict]
    return values

def make_chunk(sentence):
    for chunk in sentence:
        chunk.make()
    return sentence

#係っている2つの文節をうけとり、chunk1に名詞、chunk2に動詞があればTrueを返す。
def check_dependency(chunk1, chunk2):
    flag1, flag2 = False, False
    for morph in chunk1.morphs:
        if morph.pos == '名詞':
            flag1 = True
    for morph in chunk2.morphs:
        if morph.pos == '動詞':
            flag2 = True
    return flag1 and flag2

def print_dependency(sentence):
    sentence = make_chunk(sentence)
    for i in range(len(sentence)):
        dst = sentence[i].dst
        if dst == -1: continue
        if check_dependency(sentence[i], sentence[dst]):
            print("[{}]{}\t[{}]{}".format(sentence[i].idx\
                    ,sentence[i].chunk, sentence[dst].idx, sentence[dst].chunk))
        else: pass
        

def main():
    text = text_processing(fname)
    sentences = sentence_list(text)
    cnt = 0
    for sentence in sentences:
        cnt += 1
        print("#############sentence{}####################".format(cnt))
        print_dependency(sentence)
        if cnt == 12: break

if __name__ == "__main__":
    main()
