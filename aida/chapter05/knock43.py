from knock40 import Morph
from knock41 import Chunk, read_file
from knock42 import extract_phrases

def pos_in_phrase(pos, chunk):
    """ target pos in phrase or not
    
    :param pos: target pos
    :param chunk: chunk in sentence

    :return is_pos_in_phrase: bool, target pos in phrase or not
    """
    pos_list = [morph.pos for morph in chunk.morphs]
    is_pos_in_phrase = pos in pos_list

    return is_pos_in_phrase

if __name__ == '__main__':
    doc = read_file()
    for sentence in doc:
        for chunk_src in sentence:
            chunk_dst = sentence[chunk_src.dst]
            if pos_in_phrase('名詞', chunk_src) \
                    and pos_in_phrase('動詞', chunk_dst):
                phrases = extract_phrases(chunk_src, sentence)
                print(phrases)

