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

    def to_str(self):
        chunk = ''
        for morph in self.morphs:
            # 記号を除去
            if morph.pos == "記号": continue
            chunk += morph.surface
        self.chunk = chunk

    def print(self):
        print("[文節番号:{}] {}\t[係り先:{}]\t[係り元{}]".format(self.idx, self.chunk\
                ,self.dst, self.srcs))

class Verb_data:
    def __init__(self, base, srcs):
        self.base = base
        self.srcs = srcs


def text_processing(file):
    with open(file) as f:
        text = f.read()
        # 段落のための改行（EOSが二つ連続で出現しているところ）を消す
        c = re.compile(r'EOS\nEOS\n')
        text_processed = c.sub(r'EOS\n', text)
    return text_processed

def make_sentence_list(text):
    file = io.StringIO(text)
    sentences = []
    #文節番号:Chunkオブジェクトの辞書
    chunks = {}
    for line in file:
        if line == 'EOS\n':
            chunks = extract_values(chunks)
            sentence = list_to_string(chunks)
            sentences.append(sentence)
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
        #表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音
        cols = line.split('\t')
        columns = cols[1].split(',')
        morph = Morph(cols[0], columns[6], columns[0], columns[1])
        chunks[idx].morphs.append(morph)
    return sentences

def extract_values(dict):
    dict = sorted(dict.items(), key = lambda x:x[0])
    values = [tuple[1] for tuple in dict]
    return values
#前までのmake_chunk()のかわり
def list_to_string(sentence):
    for chunk in sentence:
        chunk.to_str()
    return sentence

def is_post_in_src(sentence, src):
    chunk = sentence[src]
    for morph in chunk.morphs:
        if morph.pos == '助詞':
            return True
    return False


def get_post(chunk):
    #print(chunk.chunk)
    for morph in chunk.morphs:
        #文節中に複数の助詞があっても最も左の助詞だけを採用する
        if morph.pos == '助詞':
            post = morph.base
        else: pass
    return post


def extract_dependency(sentence):
    # class Verbのリスト
    Verbs = []
    for chunk in sentence:
        for morph in chunk.morphs:
            if morph.pos == '動詞':
                verb = morph.base
                srcs = chunk.srcs
                Verb = Verb_data(verb, srcs)
                Verbs.append(Verb)
                break
    #("動詞", [格パターン], [格フレーム])のタプルのリスト
    vpf = []
    for Verb in Verbs:
        #格パターンを表すリスト
        pattern = []
        #格フレームを表すリスト
        frame = []
        for src in Verb.srcs:
            if is_post_in_src(sentence, src):
                #この時点では格パターンと格フレームの順序は一緒
                post = get_post(sentence[src])
                pattern.append(post)
                frame.append(sentence[src].chunk)
        verb = Verb.base
        vpf.append((verb, pattern, frame))
    return vpf


def print_pattern(sentence):
    vpf = extract_dependency(sentence)
    for item in vpf:
        verb = item[0]
        # 係る助詞が複数ある場合
        if len(item[1]) >= 2:
            #("動詞", [格パターン], [格フレーム])のタプルのリスト
            #格パターンの順序に沿って格フレームのほうもソートする
            c = zip(item[1], item[2])
            posts_sorted, chunks_sorted = zip(*sorted(c))
            # patternは格のリスト
            # スペース区切りで助詞を結合する
            pattern = ' '.join(posts_sorted)
            # スペース区切りで文節を結合する
            frame = ' '.join(chunks_sorted)
        elif len(item[1]) == 1:
            #item[1][0]は要素が一個しかない配列だからe.g)['あ']
            pattern = item[1][0]
            frame = item[2][0]
        # 係る助詞がない場合、出力されない
        else: continue
        print(f"{verb}\t{pattern}\t{frame}")


def main():
    text = text_processing(fname)
    sentences = make_sentence_list(text)
    cnt = 0
    for sentence in sentences:
        cnt += 1
        print("#############sentence{}####################".format(cnt))
        print_pattern(sentence)
        if cnt == 40: break


if __name__ == "__main__":
    main()

'''
sentence34（ジョンマッカーシー～）
'''
