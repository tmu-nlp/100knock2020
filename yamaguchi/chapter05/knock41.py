def main():
    class Morph:
        def __init__(self, surface, base, pos, pos1):
            self.surface = surface
            self.base = base
            self.pos = pos
            self.pos1 = pos1

        def show(self):
            print(self.surface, self.base, self.pos, self.pos1)

    class Chunk:
        def __init__(self, sentence_id, chunk_id, dst, srcs):
            self.sentence_id = sentence_id
            self.chunk_id = chunk_id
            self.morphs = []
            self.dst = dst
            self.srcs = srcs

        def show_morphs(self):
            morphs = ''
            for morph in self.morphs:
                morphs += morph.surface
            print("morphs:", morphs)

        def show_chunk_id(self):
            print("==========")
            print("chunk_id:", self.chunk_id)

        def show_sentence_id(self):
            if (self.chunk_id == 0):
                print("====================")
                print("sentence_id:", self.sentence_id)

        def show_dst(self):
            print("dst:", self.dst)

        def show_srcs(self):
            print("srcs:", self.srcs[self.chunk_id])

    path = 'ai.ja.txt.parsed'
    with open(path, encoding="utf-8") as f:
        text = f.read().split('\n')
    result = []
    morphs = []
    chunks = []
    srcs = [[]]
    chunk = None
    sentence_id = 0
    chunk_id = 0

    for line in text[:-1]:
        if line == 'EOS':
            result.append(morphs)
            morphs = []
            sentence_id += 1
            chunk_id = 0
            srcs = [[]]

        elif line[0] == '*':
            if chunk:
                chunks.append(chunk)
            dst = int(line.split()[2][:-1])
            diff = dst + 1 - len(srcs)
            ex = [[] for _ in range(diff)]
            srcs.extend(ex)
            if dst != -1:
                srcs[dst].append(chunk_id)
            chunk = Chunk(sentence_id, chunk_id, dst, srcs)
            chunk_id += 1

        else:
            ls = line.split('\t')
            d = {}
            tmp = ls[1].split(',')
            morph = Morph(ls[0], tmp[6], tmp[0], tmp[1])
            morphs.append(morph)
            chunk.morphs.append(morph)

    else:
        chunks.append(chunk)
    for i, chunk in enumerate(chunks):
        if chunk.sentence_id == 7:
            chunk.show_sentence_id()
            chunk.show_chunk_id()
            chunk.show_dst()
            chunk.show_srcs()
            chunk.show_morphs()

if __name__ == '__main__':
    main()