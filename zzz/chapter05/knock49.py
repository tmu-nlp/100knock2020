from zzz.chapter05.knock41 import Chunk, load_chunk_from_file


def find_common_parent(chunk_list, index1, index2):
    visited = [index1, index2]
    node1 = chunk_list[index1]
    node2 = chunk_list[index2]
    while node1 is not None:
        if node1.dst == -1:
            node1 = None
        else:
            index1 = node1.dst
            if index1 in visited:
                return index1
            visited.append(index1)
            node1 = chunk_list[index1]
    while node2 is not None:
        if node2.dst == -1:
            node2 = None
        else:
            index2 = node2.dst
            if index2 in visited:
                return index2
            visited.append(index2)
            node2 = chunk_list[index2]
    return 0


if __name__ == '__main__':
    with open('ai.ja.txt.parsed') as file:
        chunks = load_chunk_from_file(file)

        for sentence in chunks:
            for i in range(len(sentence)):
                for j in range(i + 1, len(sentence)):
                    chunk_i = sentence[i]
                    chunk_j = sentence[j]

                    phrase_i = ''.join(['X' if m.pos == '名詞' else m.surface for m in chunk_i.morphs if m.pos != '記号'])
                    # phrase_i = ''.join([m.surface for m in chunk_i.morphs if m.pos != '記号'])
                    while 'XX' in phrase_i:
                        phrase_i = phrase_i.replace('XX', 'X')
                    phrase_j = ''.join(['Y' if m.pos == '名詞' else m.surface for m in chunk_j.morphs if m.pos != '記号'])
                    # phrase_j = ''.join([m.surface for m in chunk_j.morphs if m.pos != '記号'])
                    while 'YY' in phrase_j:
                        phrase_j = phrase_j.replace('YY', 'Y')

                    parent_index = find_common_parent(sentence, i, j)
                    if parent_index == j:
                        print(phrase_i, '->', end=' ')
                        node = sentence[chunk_i.dst]
                        while node != chunk_j:
                            print(''.join([m.surface for m in node.morphs if m.pos != '記号']), '->', end=' ')
                            node = sentence[node.dst]
                        print(phrase_j)
                    else:
                        print(phrase_i, end=' ')
                        node = sentence[chunk_i.dst]
                        while node != sentence[parent_index]:
                            print('->', ''.join([m.surface for m in node.morphs if m.pos != '記号']), '->', end=' ')
                            node = sentence[node.dst]
                        print('|', end=' ')

                        print(phrase_j, end=' ')
                        node = sentence[chunk_j.dst]
                        while node != sentence[parent_index]:
                            print('->', ''.join([m.surface for m in node.morphs if m.pos != '記号']), end=' ')
                            node = sentence[node.dst]
                        print('|', end=' ')

                        print(''.join([m.surface for m in sentence[parent_index].morphs if m.pos != '記号']))
            print('EOS')