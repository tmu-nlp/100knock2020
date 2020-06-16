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
                noun_aux = []
                for src in chunk.srcs:
                    src_chunk = sentence[src]
                    pre_word = ''
                    for morph in src_chunk.morphs:
                        if morph.pos == '助詞':
                            auxiliaries.append(morph.surface)
                            noun_aux.append(pre_word + morph.surface)
                        pre_word = morph.surface
                noun_aux.sort(key=dict(zip(noun_aux, auxiliaries)).get)
                auxiliaries.sort()

                print(verb, ' '.join(auxiliaries), ' '.join(noun_aux))