from zzz.chapter05.knock40 import Morph
from zzz.chapter05.knock41 import Chunk, load_chunk_from_file

if __name__ == '__main__':
    with open('ai.ja.txt.parsed') as file:
        chunks = load_chunk_from_file(file)
        for sentence in chunks:
            for chunk in sentence:
                if chunk.dst != -1:
                    for morph in chunk.morphs:
                        if morph.pos != '記号':
                            print(morph.surface, end='')
                    print('\t', end='')
                    dst_chunk = sentence[chunk.dst]
                    for morph in dst_chunk.morphs:
                        if morph.pos != '記号':
                            print(morph.surface, end='')
                # for src in chunk.srcs:
                #     temp_chunk = sentence[src]
                #     for morph in temp_chunk.morphs:
                #         if morph.pos != '記号':
                #             print(morph.surface, end='')
                #     print('\t', end='')
                print()
            print('EOS')