# 48. 名詞から根へのパスの抽出
# 文中のすべての名詞を含む文節に対し，その文節から構文木の根に至るパスを抽出せよ．
# ただし，構文木上のパスは以下の仕様を満たすものとする．
# 各文節は（表層形の）形態素列で表現する
# パスの開始文節から終了文節に至るまで，各文節の表現を” -> “で連結する

from knock41 import getdata

def dfs(sentence, chunk, result):
    if chunk.dst == -1:
        return result
    result.append(sentence[chunk.dst])
    return dfs(sentence, sentence[chunk.dst], result)

if __name__ == "__main__":
    cabocha_file_path = "ai.ja.split.txt.parsed"
    document = getdata(cabocha_file_path)
    for sentence in document:
        for chunk in sentence:
            noun_flag = False
            for morph in chunk.morphs:
                if morph.pos == "名詞":
                    noun_flag = True
            if not noun_flag:
                continue
            result = dfs(sentence, chunk, [chunk])
            ans = []
            for chunk in result:
                surface = ""
                for morph in chunk.morphs:
                    surface += morph.surface
                ans.append(surface)
            ans = " -> ".join(ans)
            print(ans)
