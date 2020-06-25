from pprint import pprint

from knock40 import Morph
from knock41 import Chunk, read_file
from knock46 import extract_phrase

def convert(sentence):
    pl, nl = [], [chunk for chunk in sentence if '名詞' in [morph.pos for morph in chunk.morphs]]
    for i in range(len(nl) - 1):
        st1 = [''.join([morph.surface if morph.pos != '名詞' else 'X' for morph in nl[i].morphs])]
        for e in nl[i + 1:]:
            dst, p = nl[i].dst, []
            st2 = [''.join([morph.surface if morph.pos != '名詞' else 'Y' for morph in e.morphs])]

            while int(dst) != -1 and dst != sentence.index(e):
                p.append(sentence[int(dst)])
                dst = sentence[int(dst)].dst
            if len(p) < 1 or p[-1].dst != -1:
                # X -> Y
                mid = [extract_phrase(chunk) for chunk in p]
                pl.append(st1 + mid + ['Y'])
            else:
                # X | Y -> EOS
                mid, dst = [], e.dst
                while not sentence[int(dst)] in p:
                    mid.append(extract_phrase(sentence[int(dst)]))
                    dst = sentence[int(dst)].dst
                ed = [extract_phrase(sentence[int(dst)])]
                pl.append([st1, st2 + mid, ed])
    return pl

if __name__ == '__main__':
    doc = read_file()
    for sentence in doc:
        pl = (convert(sentence))
        for p in pl:
            print(p)
            if isinstance(p[0], str):
                print(' -> '.join(p))
            else:
                print(p[0][0], ' -> '.join(p[1]), p[2][0], sep=' | ')

