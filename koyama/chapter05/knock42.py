# 42. 係り元と係り先の文節の表示
# 係り元の文節と係り先の文節のテキストをタブ区切り形式ですべて抽出せよ．
# ただし，句読点などの記号は出力しないようにせよ．

from knock41 import getdata

if __name__ == "__main__":
    cabocha_file_path = "ai.ja.split.txt.parsed"
    document = getdata(cabocha_file_path)
    for sentence in document:
        for chunk in sentence:
            src = ""
            for morph in chunk.morphs:
                if morph.pos != "記号":
                    src += morph.surface
            trg = ""
            for morph in sentence[chunk.dst].morphs:
                if morph.pos != "記号":
                    trg += morph.surface
            if chunk.dst == -1:
                print(f"{src}\t@@@") # 係り先がなかったら@@@と表示する
            else:
                print(f"{src}\t{trg}")
