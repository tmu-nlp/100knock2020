
def define_morph_dictionary(surface, base, pos, pos1):
    morph = {}
    morph['surface'] = surface
    morph['base'] = base
    morph['pos'] = pos
    morph['pos1'] = pos1
    return morph

def read_file(file_path='./neko.txt.mecab'):
    """ read file tokenized
    """
    doc = []
    with open(file_path) as fp:
        sentence = []
        for line in fp:
            line = line.split('\t')
            if line[0] == 'EOS\n':
                if len(sentence) > 0:
                    doc.append(sentence)
                    sentence = []
            else:
                morphes = line[1].split(',')
                surface = line[0]
                base = morphes[6]
                pos = morphes[0]
                pos1 = morphes[1]
                morph = define_morph_dictionary(surface, base, pos, pos1)
                sentence.append(morph)

    return doc

if __name__ == '__main__':
    doc = read_file()
