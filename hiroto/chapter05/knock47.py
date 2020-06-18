#pos:名詞, pos1:サ変接続
from knock46 import *
fname = "ai.ja.txt.parsed"

class VP_data:
    def __init__(self, VP_str, srcs):
        self.VP = VP_str
        self.srcs = srcs


def is_verb_in_chunk(chunk):
    for morph in chunk.morphs:
        if morph.pos == '動詞':
            return True
    return False

def get_VP(chunk, NP, idx):
    for morph in chunk.morphs:
        if morph.pos == '動詞':
            #基本形とってくる
            verb = morph.base
            #"サ変接続＋を＋動詞の基本形"をつくる
            VP = NP + verb
            #係り元をとってくる
            #remove()を使うので，copy()を使う
            srcs = chunk.srcs.copy()
            #一旦，"サ変接続＋を"の文節を係り元から削除
            srcs.remove(idx)
            VP = VP_data(VP, srcs)
            return VP

def extract_dependency(sentence):
    # class Verbのリスト
    VPs = []
    dst = -1
    for chunk in sentence:
        i = 0
        for morph in chunk.morphs:
            #インデックスが要素のインデックスをこえたら
            if i+1 == len(chunk.morphs):
                break
            next_morph = chunk.morphs[i + 1]
            #"サ変接続＋を"かどうか
            if morph.pos == '名詞' and morph.pos1 == 'サ変接続'\
            and next_morph.base == 'を':
                dst = chunk.dst
                idx = chunk.idx
                NP = morph.surface + 'を'
                if is_verb_in_chunk(sentence[dst]):
                    VP = get_VP(sentence[dst], NP, idx)
                    VPs.append(VP)
                    break
                else: pass
            else: pass
            i += 1
    #(VP, [格パターン], [格フレーム])のタプルのリスト
    vpfs = []
    for VP in VPs:
        #格パターンを表すリスト
        pattern = []
        #格フレームを表すリスト
        frame = []
        for src in VP.srcs:
            #助詞があるかどうか
            if is_post_in_src(sentence, src):
                #この時点では格パターンと格フレームの順序は一緒
                post = get_post(sentence[src])
                pattern.append(post)
                frame.append(sentence[src].chunk)
        VP = VP.VP
        vpfs.append((VP, pattern, frame))
    return vpfs


def print_mining(sentence):
    vpfs = extract_dependency(sentence)
    for vpf in vpfs:
        VP = vpf[0]
        # 係る助詞が複数ある場合
        if len(vpf[1]) >= 2:
            #(VP, [格パターン], [格フレーム])のタプルのリスト
            #格パターンの順序に沿って格フレームのほうもソートする
            c = zip(vpf[1], vpf[2])
            posts_sorted, chunks_sorted = zip(*sorted(c))
            # patternは格のリスト
            # スペース区切りで助詞を結合する
            pattern = ' '.join(posts_sorted)
            # スペース区切りで文節を結合する
            frame = ' '.join(chunks_sorted)
        elif len(vpf[1]) == 1:
            #item[1][0]は要素が一個しかない配列だからe.g)['あ']
            pattern = vpf[1][0]
            frame = vpf[2][0]
        else: continue
        print(f"{VP}\t{pattern}\t{frame}")

def out_to_file(sentences):
    with open('mining.txt', mode = 'w') as f:
        for sentence in sentences:
            vpfs = extract_dependency(sentence)
            for vpf in vpfs:
                VP = vpf[0]
                # 係る助詞が複数ある場合
                if len(vpf[1]) >= 2:
                    #(VP, [格パターン], [格フレーム])のタプルのリスト
                    #格パターンの順序に沿って格フレームのほうもソートする
                    c = zip(vpf[1], vpf[2])
                    posts_sorted, chunks_sorted = zip(*sorted(c))
                    # patternは格のリスト
                    # スペース区切りで助詞を結合する
                    pattern = ' '.join(posts_sorted)
                    # スペース区切りで文節を結合する
                    frame = ' '.join(chunks_sorted)
                elif len(vpf[1]) == 1:
                    #item[1][0]は要素が一個しかない配列だからe.g)['あ']
                    pattern = vpf[1][0]
                    frame = vpf[2][0]
                else: continue
                line = f'{VP}\t{pattern}\t{frame}\n'
                f.write(line)



def main():
    text = text_processing(fname)
    sentences = make_sentence_list(text)
    cnt = 0
    for sentence in sentences:
        cnt += 1
        print("#############sentence{}####################".format(cnt))
        print_mining(sentence)
        if cnt == 13: break

    #out_to_file(sentences)


if __name__ == "__main__":
    main()

#コーパス中で頻出する述語
#cut -f1  mining.txt | sort | uniq --count | sort -n --reverse | less
#cat case.txt | sort | uniq -count | sort -n --reverse | less

'''
sentence13
'''
