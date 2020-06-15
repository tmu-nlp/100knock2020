import re
from zzz.chapter05.knock40 import Morph

class Chunk:
    def __init__(self,
                 morphs: list = [],
                 dst: int = 0,
                 srcs: list = []):
        self.morphs = morphs
        self.dst = dst
        self.srcs = srcs

    def append_morph(self, morph: Morph):
        self.morphs.append(morph)

    def append_src(self, src):
        self.srcs.append(src)


def load_chunk_from_file(file):
    chunks = []
    cache = ''.join([line for line in file])
    cache = cache.replace('EOS\nEOS', 'EOS')
    cache = cache.split('EOS')
    for sentence in cache:
        temp_chunks = []
        temp_morphs = []
        current_index = 0
        for item in list(filter(lambda x: x != '', sentence.split('\n'))):
            if item[0] == '*':
                if len(temp_chunks) > 0:
                    temp_chunks[current_index].morphs += temp_morphs
                    temp_morphs = []

                chunk = item.split(' ')
                current_index = int(chunk[1])
                dst = int(chunk[2][:-1])
                # srcs = []

                if len(temp_chunks) <= current_index:
                    temp_chunks.append(Chunk([], 0, []))
                temp_chunks[current_index].dst = dst

                if dst != -1:
                    while len(temp_chunks) <= dst:
                        temp_chunks.append(Chunk([], 0, []))
                    temp_chunks[dst].append_src(current_index)

            else:
                morph = re.split(r'\t|,|\n', item)
                temp_morphs.append(Morph(morph[0], morph[7], morph[1], morph[2]))
        if len(temp_chunks) > 0:
            temp_chunks[-1].morphs += temp_morphs

        chunks.append(temp_chunks)
    return chunks


if __name__ == '__main__':
    with open('ai.ja.txt.parsed') as file:
        chunks = load_chunk_from_file(file)
