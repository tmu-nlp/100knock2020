# 49. 名詞間の係り受けパスの抽出
# 文中のすべての名詞句のペアを結ぶ最短係り受けパスを抽出せよ．
# ただし，名詞句ペアの文節番号がiとj（i<j）のとき，係り受けパスは以下の仕様を満たすものとする．
# 問題48と同様に，パスは開始文節から終了文節に至るまでの各文節の表現（表層形の形態素列）を” -> “で連結して表現する
# 文節iとjに含まれる名詞句はそれぞれ，XとYに置換する
# また，係り受けパスの形状は，以下の2通りが考えられる．
# 文節iから構文木の根に至る経路上に文節jが存在する場合: 文節iから文節jのパスを表示
# 上記以外で，文節iと文節jから構文木の根に至る経路上で共通の文節kで交わる場合: 文節iから文節kに至る直前のパスと文節jから文節kに至る直前までのパス，文節kの内容を” | “で連結して表示

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
        for i in range(len(sentence)):
            noun_flag1 = False
            for morph in sentence[i].morphs:
                if morph.pos == "名詞":
                    noun_flag1 = True
            if not noun_flag1:
                continue
            for j in range(i + 1, len(sentence)):
                noun_flag2 = False
                for morph in sentence[j].morphs:
                    if morph.pos == "名詞":
                        noun_flag2 = True
                if not noun_flag2:
                    continue
                result1 = dfs(sentence, sentence[i], [sentence[i]])
                if sentence[j] in result1:
                    ans = []
                    for chunk in result1:
                        ans.append(chunk)
                        if sentence[j] == chunk:
                            break
                    ans_chunk = []
                    surface = ""
                    first_X = True
                    for morph in ans[0].morphs:
                        if morph.pos == "名詞": 
                            if first_X:
                                surface += "X"
                                first_X = False
                        else:
                            surface += morph.surface
                    ans_chunk.append(surface)
                    for chunk in ans[1:len(ans) - 1]:
                        surface = ""
                        for morph in chunk.morphs:
                            surface += morph.surface
                        ans_chunk.append(surface)
                    surface = ""
                    first_Y = True
                    for morph in ans[-1].morphs:
                        if morph.pos == "名詞":
                            if first_Y:
                                surface += "Y"
                                first_Y = False
                        else:
                            surface += morph.surface
                    ans_chunk.append(surface)
                    ans_chunk = " -> ".join(ans_chunk)
                    print(f"{ans_chunk}")
                else:
                    result2 = dfs(sentence, sentence[j], [sentence[j]])
                    cross_point = ""
                    for chunk1 in result1:
                        for chunk2 in result2:
                            if chunk1 == chunk2:
                                cross_point = chunk1
                    ans1_chunk = []
                    if result1[0] != cross_point:
                        surface = ""
                        first_X = True
                        for morph in result1[0].morphs:
                            if morph.pos == "名詞":
                                if first_X:
                                    surface += "X"
                                    first_X = False
                            else:
                                surface += morph.surface
                        ans1_chunk.append(surface)
                    for chunk1 in result1[1:]:
                        if chunk1 == cross_point:
                            break
                        surface = ""
                        for morph in chunk1.morphs:
                            surface += morph.surface
                        ans1_chunk.append(surface)
                    ans2_chunk = []
                    if result2[0] != cross_point:
                        surface = ""
                        first_Y = True
                        for morph in result2[0].morphs:
                            if morph.pos == "名詞": 
                                if first_Y:
                                    surface += "Y"
                                    first_Y = False
                            else:
                                surface += morph.surface
                        ans2_chunk.append(surface)
                    for chunk2 in result2[1:]:
                        if chunk2 == cross_point:
                            break
                        surface = ""
                        for morph in chunk2.morphs:
                            surface += morph.surface
                        ans2_chunk.append(surface)
                    cross_surface = ""
                    for morph in cross_point.morphs:
                        cross_surface += morph.surface
                    ans1_chunk = " -> ".join(ans1_chunk)
                    ans2_chunk = " -> ".join(ans2_chunk)
                    print(f"{ans1_chunk} | {ans2_chunk} | {cross_surface}")


                # ans1 = []
                # for chunk in result1:
                #     surface = ""
                #     for morph in chunk.morphs:
                #         surface += morph.surface
                #     ans1.append(surface)
                # ans1 = " -> ".join(ans1)
                # print(ans1)
                # ans2 = []
                # for chunk in result2:
                #     surface = ""
                #     for morph in chunk.morphs:
                #         surface += morph.surface
                #     ans2.append(surface)
                # ans2 = " -> ".join(ans2)
                # print(ans2)
