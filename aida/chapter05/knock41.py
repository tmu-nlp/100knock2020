from knock40 import Morph

class Chunk():
    def __init__(self, dst, srcs):
        self.morphs = []
        self.dst = dst
        self.srcs = srcs

def read_file(file_path='./ai.ja.txt.parsed'):
    doc = []
    chunk = Chunk(None, None)
    with open(file_path) as fp:
        sentence = []
        for line in fp:
            line = line.strip()
            if line[0] == '*':
                if len(chunk.morphs) > 0:
                    sentence.append(chunk)
                # define chunk
                items = line.split()
                srcs = int(items[1])
                dst = int(items[2][:-1])
                chunk = Chunk(dst, srcs)

            elif line[:3] == 'EOS':
                if len(chunk.morphs) > 0:
                    sentence.append(chunk)
                if len(sentence) == 0:
                    continue
                chunk = Chunk(None, None)
                doc.append(sentence)
                sentence = []
            else:
                # define morph
                word_parts = line.split('\t')
                parts = word_parts[1].split(',')
                word = word_parts[0]
                surface = parts[6]
                pos = parts[0]
                pos1 = parts[1]
                morph = Morph(word, surface, pos, pos1)
                #sentence.append(morph)
                chunk.morphs.append(morph)
    return doc

if __name__ == '__main__':
    doc = read_file()
    # test
    sentence = doc[1]
    for chunk in sentence:
        print(f'srcs: {chunk.srcs} dst: {chunk.dst}')
        morphs_surface = [morph.surface for morph in chunk.morphs]
        print(f'morphs_surface: {" ".join(morphs_surface)}')

