# 46. 動詞の格フレーム情報の抽出
# 45のプログラムを改変し，述語と格パターンに続けて項（述語に係っている文節そのもの）をタブ区切り形式で出力せよ．
# 45の仕様に加えて，以下の仕様を満たすようにせよ．
# 項は述語に係っている文節の単語列とする（末尾の助詞を取り除く必要はない）
# 述語に係る文節が複数あるときは，助詞と同一の基準・順序でスペース区切りで並べる

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
            particles = [] # (助詞, 節)というタブルのリストとする
            for src in chunk.srcs:
                surface = ""
                for morph in sentence[src].morphs:
                    surface += morph.surface
                for morph in sentence[src].morphs:
                    if morph.pos == "助詞":
                        particles.append((morph.base, surface))
            particles = sorted(particles)
            particle_s = []
            chunk_s = []
            for particle in particles:
                particle_s.append(particle[0])
                chunk_s.append(particle[1])
            particle_s = " ".join(particle_s)
            chunk_s = " ".join(chunk_s)
            print(f"{verb}\t{particle_s} {chunk_s}")