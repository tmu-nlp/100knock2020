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
    #idx = 0
    sentences = []
    chunks = {}
    for line in file:
        if line == 'EOS\n':
            chunks = extract_values(chunks)
            sentence = make_chunk(chunks)
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

def check_dependency(chunk1, chunk2):
    flag1, flag2 = False, False
    for morph in chunk1.morphs:
        if morph.pos == '名詞':
            flag1 = True
    for morph in chunk2.morphs:
        if morph.pos == '動詞':
            flag2 = True
    return flag1 and flag2

#文節中の助詞を取ってくる
def get_post(chunk):
    for morph in chunk.morphs:
        #最も左の助詞だけ抜き出す
        if morph.pos == '助詞':
            post = morph.base
        else: pass
    return post

#動詞にかかる文節中に助詞があるかどうか
def is_post_in_src(sentence, src):
    chunk = sentence[src]
    for morph in chunk.morphs:
        if morph.pos == '助詞':
            return True
    return False

def dependency(sentence):
    # class Verbのリスト
    Verbs = []
    for chunk in sentence:
        for morph in chunk.morphs:
            if morph.pos == '動詞':
                #最も左の動詞だけぬき出す
                verb = morph.base
                srcs = chunk.srcs
                Verb = Verb_data(verb, srcs)
                Verbs.append(Verb)
                break
    #("動詞", [助詞のリスト])のタプルのリスト
    verb_posts_list = []
    for Verb in Verbs:
        #動詞にかかる助詞のリスト
        posts = []
        for src in Verb.srcs:
            if is_post_in_src(sentence, src):
                post = get_post(sentence[src])
                posts.append(post)
        verb = Verb.base
        verb_posts_list.append((verb, posts))
    return verb_posts_list

def print_pattern(sentence):
    list = dependency(sentence)
    for item in list:
        # 係る助詞が複数ある場合
        if len(item[1]) >= 2:
            # 助詞をソートする
            posts_sorted = sorted(item[1])
            # caseは格のリスト
            case = ' '.join(posts_sorted)
        elif len(item[1]) == 1:
            #item[1][0]は要素が一個しかない配列だからe.g)['あ']
            case = item[1][0]
        else: continue
        print("{}\t{}".format(item[0], case))


def write_pattern(sentences):
    with open('case.txt', mode = 'w') as f:
        for sentence in sentences:
            list = dependency(sentence)
            for item in list:
                # 係る助詞が複数ある場合
                if len(item[1]) >= 2:
                    # 助詞をソートする
                    posts_sorted = sorted(item[1])
                    # caseは格のリスト
                    case = ' '.join(posts_sorted)
                elif len(item[1]) == 1:
                    #item[1][0]は要素が一個しかない配列だからe.g)['あ']
                    case = item[1][0]
                else: continue
                str = '{}\t{}\n'.format(item[0], case)
                f.write(str)


def main():
    text = text_processing(fname)
    sentences = make_sentence_list(text)
    # 8番目の文
    cnt = 0
    for sentence in sentences:
        cnt += 1
        print("#############sentence{}####################".format(cnt))
        print_pattern(sentence)
        if cnt == 34: break

    write_pattern(sentences)

if __name__ == "__main__":
    main()

# cat case.txt | sort | uniq -c | sort -n --reverse | less 
# cat case.txt | grep "^行う\s" | sort | uniq --count | sort -n --reverse | less
# cat case.txt | grep "^なる\s" | sort | uniq --count | sort -n --reverse | less
# cat case.txt | grep "^与える\s" | sort | uniq --count | sort -n --reverse | less

'''
case.txtの76行目(ジョンマッカーシー…)
'''

'''
      8 行う    を
      2 行う    は を
      1 行う    まで を
      1 行う    は を をめぐって
      1 行う    は は は
      1 行う    に を を
      1 行う    に まで を
      1 行う    に により を
      1 行う    に
      1 行う    で を
      1 行う    で に を
      1 行う    て に を
      1 行う    て に
      1 行う    が は
      1 行う    が で は
      1 行う    が て に は
      1 行う    から
'''

'''
      3 なる    に は
      3 なる    が と
      2 なる    に
      2 なる    と
      1 なる    も
      1 なる    は も
      1 なる    は は
      1 なる    に は は
      1 なる    と など は
      1 なる    で は
      1 なる    で に は
      1 なる    て として に は
      1 なる    が にとって は
      1 なる    が に は
      1 なる    が に
      1 なる    が で と に は は
      1 なる    が が と
      1 なる    から で と
      1 なる    から が て と は ば
'''

'''
      2 与える  が に
      1 与える  に は を
'''
