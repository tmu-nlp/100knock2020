from graphviz import Digraph

from knock40 import Morph
from knock41 import Chunk, read_file

def extract_phrases(chunk_srcs, sentence):
    """ obtain srcs and dst phrases

    :param chunk_srcs: target chunk
    :param sentence: list, each sentence includes chunks
    :return phrases: list, srcs and dst phrases
    """
    chunk_dst = sentence[chunk_srcs.dst]
    phrase_srcs = ''.join([morph.surface for morph in chunk_srcs.morphs if morph.pos != '記号'])
    phrase_dst = ''.join([morph.surface for morph in chunk_dst.morphs if morph.pos != '記号'])
    return phrase_srcs, phrase_dst

if __name__ == '__main__':
    G = Digraph(format='png')
    G.attr("node", shape="square", style="filled")

    doc = read_file()
    sentence = doc[7]
    for chunk in sentence:
        if chunk.dst == -1:
            break
        phrase_srcs, phrase_dst = extract_phrases(chunk, sentence)
        G.edge(phrase_srcs, phrase_dst)

    G.render('graphs')


