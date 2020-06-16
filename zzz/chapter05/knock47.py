from zzz.chapter05.knock41 import Chunk, load_chunk_from_file

if __name__ == '__main__':
    with open('ai.ja.txt.parsed') as file:
        chunks = load_chunk_from_file(file)
        for sentence in chunks:
            for chunk in sentence:
                verb = ''
                for morph in chunk.morphs:
                    if morph.pos == '動詞':
                        verb = morph.base
                    break
                # else:
                #     continue
                if len(verb) == 0:
                    continue

                if len(chunk.srcs) > 0:
                    temp_chunk = sentence[chunk.srcs[-1]]
                    if len(temp_chunk.morphs) > 1 and \
                            temp_chunk.morphs[-1].surface == 'を' and \
                            temp_chunk.morphs[-2].pos1 == 'サ変接続':
                        verb = temp_chunk.morphs[-2].surface + temp_chunk.morphs[-1].surface + verb
                        auxiliaries = []
                        noun_aux = []
                        for src in chunk.srcs[:-1]:
                            src_chunk = sentence[src]
                            pre_word = ''
                            for morph in src_chunk.morphs:
                                if morph.pos == '助詞':
                                    auxiliaries.append(morph.surface)
                                    noun_aux.append(pre_word + morph.surface)
                                pre_word = morph.surface
                        if len(auxiliaries) > 0:
                            noun_aux.sort(key=dict(zip(noun_aux, auxiliaries)).get)
                            auxiliaries.sort()

                            print(verb, ' '.join(auxiliaries), ' '.join(noun_aux))