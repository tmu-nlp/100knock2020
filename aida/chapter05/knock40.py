class Morph():
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1

def read_file(file_path='./ai.ja.txt.parsed'):
    doc = []
    with open(file_path) as fp:
        sentence = []
        for line in fp:
            line = line.strip()
            if line[0] == '*':
                continue
            elif line[:3] == 'EOS':
                if len(sentence) == 0:
                    continue
                doc.append(sentence)
                sentence = []
            else:
                word_parts = line.split('\t')
                parts = word_parts[1].split(',')
                word = word_parts[0]
                surface = parts[6]
                pos = parts[0]
                pos1 = parts[1]
                morph = Morph(word, surface, pos, pos1)
                sentence.append(morph)
    return doc

if __name__ == '__main__':
    doc = read_file()

