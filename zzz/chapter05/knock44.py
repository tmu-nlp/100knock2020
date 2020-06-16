from zzz.chapter05.knock41 import Chunk, load_chunk_from_file
from graphviz import Digraph



if __name__ == '__main__':
    with open('ai.ja.txt.parsed') as file:
        chunks = load_chunk_from_file(file)
        for (sent_index, sentence) in enumerate(chunks):
            dot = Digraph(comment='Graph for sentence {}'.format(sent_index))
            dot.node(str(-1), 'ROOT')
            for (chunk_index, chunk) in enumerate(sentence):
                morph = ''.join(m.surface for m in chunk.morphs)

                if chunk.dst != -1:
                    dst_chunk = sentence[chunk.dst]
                    dst_morph = ''.join(m.surface for m in dst_chunk.morphs)
                    dot.node(str(chunk.dst), dst_morph)

                dot.node(str(chunk_index), morph)
                dot.edge(str(chunk_index), str(chunk.dst))
                # print(dot.source)
            dot.format = 'png'
            dot.render('output-graph-{}'.format(sent_index))