from pprint import pprint

from knock40 import Morph
from knock41 import Chunk, read_file
from knock46 import extract_phrase

if __name__ == '__main__':
    doc = read_file()
    pathes = []
    for sentence in doc:
        for chunk in sentence:
            dsts = [extract_phrase(chunk)]
            while chunk.dst != -1:
                chunk = sentence[chunk.dst]
                dsts.append(extract_phrase(chunk))
            pathes.append(dsts)
    for path in pathes:
        print(' -> '.join(path))

