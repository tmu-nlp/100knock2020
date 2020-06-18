from graphviz import Digraph

from knock40 import Morph
from knock41 import Chunk, read_file

def extract_phrases(chunk_srcs, sentence):
    """ obtain srcs and dst phrases

    :param chunk_srcs: target chunk
    :param sentence: list, each sentence includes chunks
    :return phrases: list, srcs dst phrases
    """
    chunk_dst = sentence[chunk_srcs.dst]
    phrase_srcs = ''.join([morph.surface for morph in chunk_srcs.morphs if morph.pos != '記号'])
    phrase_dst = ''.join([morph.surface for morph in chunk_dst.morphs if morph.pos != '記号'])
    phrases = f'{phrase_srcs}\t{phrase_dst}'
    return phrases

if __name__ == '__main__':
    G = Digraph(format='png')
    G.attr("node", shape="square", style="filled")

    doc = read_file()
    sentence = doc[7]
    for chunk in sentence:
        phrases = extract_phrases(chunk, sentence).split('\t')
        phrase_srcs = phrases[0]
        phrase_dst = phrases[1]
        G.edge(phrase_srcs, phrase_dst)

    G.render('graphs')


