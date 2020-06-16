from zzz.chapter05.knock40 import Morph
from zzz.chapter05.knock41 import Chunk, load_chunk_from_file


def find_pos(chunk: Chunk, target: str):
    for morph in chunk.morphs:
        if morph.pos == target:
            return True
    return False


if __name__ == '__main__':
    with open('ai.ja.txt.parsed') as file:
        chunks = load_chunk_from_file(file)
        for sentence in chunks:
            for chunk in sentence:
                if chunk.dst != -1:
                    dst_chunk = sentence[chunk.dst]
                    if find_pos(chunk, '名詞') and find_pos(dst_chunk, '動詞'):
                        for morph in chunk.morphs:
                            if morph.pos != '記号':
                                print(morph.surface, end='')
                        print('\t', end='')
                        for morph in dst_chunk.morphs:
                            if morph.pos != '記号':
                                print(morph.surface, end='')
                        print()
            print('EOS')