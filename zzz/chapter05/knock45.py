from zzz.chapter05.knock41 import Chunk, load_chunk_from_file

if __name__ == '__main__':
    with open('ai.ja.txt.parsed') as file:
        chunks = load_chunk_from_file(file)
        for sentence in chunks:
            for chunk in sentence:
                verb = ''
                for morph in chunk.morphs:
                    if morph.pos == '動詞':
                        verb = morph.surface
                    break
                # else:
                #     continue
                if len(verb) == 0:
                    continue

                auxiliaries = []
                for src in chunk.srcs:
                    src_chunk = sentence[src]
                    for morph in src_chunk.morphs:
                        if morph.pos == '助詞':
                            auxiliaries.append(morph.surface)
                auxiliaries.sort()

                print(verb, ' '.join(auxiliaries))