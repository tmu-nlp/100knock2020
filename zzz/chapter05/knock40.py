import re


class Morph:
    def __init__(self,
                 surface: str = '',
                 base: str = '',
                 pos: str = '',
                 pos1: str = ''):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1


def load_morph_from_file(file):
    morphs = []
    cache = ''.join([line for line in file])
    cache = cache.replace('EOS\nEOS', 'EOS')
    cache = cache.split('EOS')
    for sentence in cache:
        temp_morphs = []
        for item in sentence.split('\n'):
            if item[0] == '*':
                morph = re.split(r'\t|,|\n', item)
                temp_morphs.append(Morph(morph[0], morph[7], morph[1], morph[2]))
        if len(temp_morphs) > 0:
            morphs.append(temp_morphs)
    return morphs


if __name__ == '__main__':
    with open('ai.ja.txt.parsed') as file:
        morphs = load_morph_from_file(file)

        for sentence in morphs:
            for morph in sentence:
                print(morph.surface, morph.pos, morph.pos1, morph.base)
            print('EOS')
