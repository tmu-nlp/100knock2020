# 43. 名詞を含む文節が動詞を含む文節に係るものを抽出
# 名詞を含む文節が，動詞を含む文節に係るとき，これらをタブ区切り形式で抽出せよ．
# ただし，句読点などの記号は出力しないようにせよ．

from knock41 import getdata

if __name__ == "__main__":
    cabocha_file_path = "ai.ja.split.txt.parsed"
    document = getdata(cabocha_file_path)
    for sentence in document:
        for chunk in sentence:
            src = ""
            noun_flag = False
            for morph in chunk.morphs:
                if morph.pos != "記号":
                    src += morph.surface
                if morph.pos == "名詞":
                    noun_flag = True
            trg = ""
            verb_flag = False
            for morph in sentence[chunk.dst].morphs:
                if morph.pos != "記号":
                    trg += morph.surface
                if morph.pos == "動詞":
                    verb_flag = True
            if noun_flag and verb_flag:
                print(f"{src}\t{trg}")