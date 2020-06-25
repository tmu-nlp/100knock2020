# 47. 機能動詞構文のマイニング
# 動詞のヲ格にサ変接続名詞が入っている場合のみに着目したい．
# 46のプログラムを以下の仕様を満たすように改変せよ．
# 「サ変接続名詞+を（助詞）」で構成される文節が動詞に係る場合のみを対象とする
# 述語は「サ変接続名詞+を+動詞の基本形」とし，文節中に複数の動詞があるときは，最左の動詞を用いる
# 述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる
# 述語に係る文節が複数ある場合は，すべての項をスペース区切りで並べる（助詞の並び順と揃えよ）

from knock41 import getdata

if __name__ == "__main__":
    cabocha_file_path = "ai.ja.split.txt.parsed"
    document = getdata(cabocha_file_path)
    for sentence in document:
        for i, chunk in enumerate(sentence):
            verb = ""
            predicate = []
            for morph in chunk.morphs:
                predicate.append(morph)
                if morph.pos == "動詞":
                    verb = morph.base
                    break
            if verb == "":
                continue
            for j in range(i - 1, -1, -1):
                if len(predicate) >= 3:
                    break
                for morph in sentence[j].morphs[::-1]:
                    if len(predicate) >= 3:
                        break
                    predicate.insert(0, morph)
            if len(predicate) <= 2:
                continue
            if not (predicate[-3].pos1 == "サ変接続" and predicate[-2].surface == "を" and predicate[-1].pos == "動詞"):
                continue
            verbs = predicate[-3].surface + predicate[-2].surface + predicate[-1].base
            particles = []
            for src in chunk.srcs:
                if sentence[src].index + 1 == chunk.index:
                    continue
                surfaces = ""
                for morph in sentence[src].morphs:
                    surfaces += morph.surface
                for morph in sentence[src].morphs:
                    if morph.pos == "助詞":
                        particles.append((morph.base, surfaces))
            particles = sorted(particles)
            particle_s = []
            chunk_s = []
            for particle in particles:
                particle_s.append(particle[0])
                chunk_s.append(particle[1])
            particle_s = " ".join(particle_s)
            chunk_s = " ".join(chunk_s)
            print(f"{verbs}\t{particle_s} {chunk_s}")