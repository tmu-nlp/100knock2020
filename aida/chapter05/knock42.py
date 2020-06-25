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
    doc = read_file()
    for sentence in doc:
        for chunk in sentence:
            phrases = extract_phrases(chunk, sentence)
            print(phrases)

