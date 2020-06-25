import re, io
from itertools import combinations as comb
from knock46 import Morph, text_processing, extract_values, list_to_string
fname = "ai.ja.txt.parsed"

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

    #文節中の名詞部分をcharacterに指定した文字で置き換える
    def rep_noun(self, character):
        chunk = ''
        temp = ''
        flag = True
        for morph in self.morphs:
            '''
            if morph.pos == '助詞':
                chunk = character + morph.surface
                break
            else: pass
            '''
            #現代技術とかだと現代、技術でそれぞれ名詞に分けられるから、XXになる
            #XまたはYが２個並ばないようにflagを使って一個だけならぶ（XXを一つのXに
            #まとめる）ようにする
            if morph.pos == '名詞' and flag:
                chunk += character
                flag = False
            elif morph.pos == '助詞':
                temp = morph.surface
            else: pass
        chunk += temp

        return chunk

def make_sentence_list(text):
    file = io.StringIO(text)
    #idx = 0
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

#名詞句のペアを作る
def make_NP_pairs(sentence):
    NP_indices = []
    for chunk in sentence:
        flag = False
        for morph in chunk.morphs:
            if morph.pos == '名詞':
                flag = True
            if morph.pos == '動詞':
                flag = False
                break
        if flag == True:
            NP_indices.append(chunk.idx)
    NP_pairs = list(comb(NP_indices, 2))
    return NP_pairs

#二つの名詞句の根までのパスを抽出する
def extract_paths(sentence, pair):
    # 名詞句i < 名詞句j
    i, j = pair
    NP1_path, NP2_path = [], []
    NP1_path.append(i)
    NP2_path.append(j)
    NP1_path = follow(sentence, i, NP1_path)
    NP2_path = follow(sentence, j, NP2_path)
    return NP1_path, NP2_path, j

#文中の一つのパスの抽出を再帰的に求める
def follow(sentence, idx, path):
    dst = sentence[idx].dst
    if dst != -1:
        #print(dst)
        path.append(dst)
        #print(path)
        idx = dst
        path = follow(sentence, idx, path)
        return path
    else: return path

#文中のパスを表示
def print_path(sentence):
    NP_pairs = make_NP_pairs(sentence)
    for pair in NP_pairs:
        NP1_path, NP2_path, j = extract_paths(sentence, pair)
        #iからjが直接繋がっている場合
        if j in NP1_path:
            idx = NP1_path.index(j)
            path = NP1_path[0:idx+1]
            one_path(path, sentence)
        else:
            #文節kで交わる場合
            k = list(set(NP1_path) & set(NP2_path))[0]
            NP1_k_idx = NP1_path.index(k)
            path_i_to_k = NP1_path[0:NP1_k_idx]
            NP2_k_idx = NP2_path.index(k)
            path_j_to_k = NP2_path[0:NP2_k_idx]
            two_paths(path_i_to_k, path_j_to_k, k, sentence)


def one_path(path, sentence):
    cnt = 0
    length = len(path)
    for idx in path:
        if cnt == 0:
            chunk = sentence[idx].rep_noun('X')
            print(chunk, end = ' -> ')
        #最後のインデックスを指しているとき
        elif cnt == length - 1:
            chunk = sentence[idx].rep_noun('Y')
            print(chunk)
        else:
            print(sentence[idx].chunk, end = ' -> ')
        cnt += 1

def two_paths(path_i_to_k, path_j_to_k, k, sentence):
    #iの文節からkの一つ前までの文節のパスを表示する
    cnt = 0
    length = len(path_i_to_k)
    for idx in path_i_to_k:
        if cnt == 0:
            chunk = sentence[idx].rep_noun('X')
            print(chunk, end = '')
            if length == 1:
                print(' | ', end = '')
            else: print(' -> ', end = '')
        #最後のインデックスを指しているとき
        elif cnt == length - 1:
            print(sentence[idx].chunk, end = ' | ')
        else:
            print(sentence[idx].chunk, end = ' -> ')
        cnt += 1
    #jの文節からkの一つ前までの文節のパスを表示する
    cnt = 0
    length = len(path_j_to_k)
    for idx in path_j_to_k:
        if cnt == 0:
            chunk = sentence[idx].rep_noun('Y')
            print(chunk, end = '')
            if length == 1:
                print(' | ', end = '')
            else: print(' -> ', end = '')
        #最後のインデックスを指しているとき
        elif cnt == length - 1:
            print(sentence[idx].chunk, end = ' | ')
        else:
            print(sentence[idx].chunk, end = ' -> ')
        cnt += 1
    #kの文節を表示する
    print(sentence[k].chunk)


def main():
    text = text_processing(fname)
    sentences = make_sentence_list(text)
    cnt = 0
    for sentence in sentences:
        cnt += 1
        print(f"#############sentence{cnt}####################")
        print_path(sentence)
        if cnt == 32: break


if __name__ == "__main__":
    main()
