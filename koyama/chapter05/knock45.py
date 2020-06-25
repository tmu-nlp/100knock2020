# 45. 動詞の格パターンの抽出
# 今回用いている文章をコーパスと見なし，日本語の述語が取りうる格を調査したい．
# 動詞を述語，動詞に係っている文節の助詞を格と考え，述語と格をタブ区切り形式で出力せよ．
# ただし，出力は以下の仕様を満たすようにせよ．
# 動詞を含む文節において，最左の動詞の基本形を述語とする
# 述語に係る助詞を格とする
# 述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる

from knock41 import getdata

if __name__ == "__main__":
    cabocha_file_path = "ai.ja.split.txt.parsed"
    document = getdata(cabocha_file_path)
    for sentence in document:
        for chunk in sentence:
            verb = ""
            for morph in chunk.morphs:
                if morph.pos == "動詞":
                    verb = morph.base
                    break # 最左のものを述語とするためにbreakする
            if verb == "":
                continue
            particles = []
            for src in chunk.srcs:
                for morph in sentence[src].morphs:
                    if morph.pos == "助詞":
                        particles.append(morph.base)
            particles = sorted(particles)
            particles = " ".join(particles)
            print(f"{verb}\t{particles}")