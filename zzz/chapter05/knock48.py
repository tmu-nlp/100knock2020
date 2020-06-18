from zzz.chapter05.knock41 import Chunk, load_chunk_from_file

if __name__ == '__main__':
    with open('ai.ja.txt.parsed') as file:
        chunks = load_chunk_from_file(file)

        for sentence in chunks:
            for chunk in sentence:
                node = chunk
                while node is not None:
                    if node.dst != -1:
                        print(''.join([m.surface for m in node.morphs if m.pos != '記号']), '->', end=' ')
                        node = sentence[node.dst]
                    else:
                        print(''.join([m.surface for m in node.morphs if m.pos != '記号']))
                        node = None
            print('EOS')