from knock46 import Morph, Chunk, text_processing, make_sentence_list\
, extract_values, list_to_string
fname = "ai.ja.txt.parsed"

#文中の全てのパスの抽出
def extract_paths(sentence):
    paths = []
    for chunk in sentence:
        path = []
        for morph in chunk.morphs:
            if morph.pos == '名詞' and chunk.dst != -1:
                idx = chunk.idx
                path.append(idx)
                path = follow(sentence, idx, path)
                paths.append(path)
                break
            else: pass
    return paths

#文中の一つのパスの抽出を再帰的に求める
def follow(sentence, idx, path):
    dst = sentence[idx].dst
    if dst != -1:
        path.append(dst)
        idx = dst
        path = follow(sentence, idx, path)
        return path
    else: return path

#文中のパスを表示
def print_path(sentence):
    paths = extract_paths(sentence)
    for path in paths:
        cnt = 0
        length = len(path)
        for idx in path:
            print(f'{sentence[idx].chunk}', end = '')
            #最後のインデックスを指しているとき
            if cnt == length - 1: pass
            else: print(' -> ', end = '')
            cnt += 1
        print()


def main():
    text = text_processing(fname)
    sentences = make_sentence_list(text)
    cnt = 0
    for sentence in sentences:
        cnt += 1
        print(f"#############sentence{cnt}####################")
        print_path(sentence)
        if cnt == 40: break


if __name__ == "__main__":
    main()

'''
ai.ja.txt.parsedの1488行目のAIが名詞だと判断されない==>辞書の違い
* 1 2D 0/2 0.550795
A	記号,アルファベット,*,*,*,*,A,エイ,エイ,,
I	記号,アルファベット,*,*,*,*,I,アイ,アイ,,
に関する	助詞,格助詞,連語,*,*,*,に関する,ニカンスル,ニカンスル,,
'''
