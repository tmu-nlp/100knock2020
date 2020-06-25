from pprint import pprint

from knock40 import Morph
from knock41 import Chunk, read_file
from knock43 import pos_in_phrase
from knock45 import obtain_word

def extract_phrase(chunk_srcs):
    surfaces = [morph.surface for morph in chunk_srcs.morphs if morph.pos != '記号']
    return ''.join(surfaces)

if __name__ == '__main__':
    doc = read_file()
    verb_pps_all = []
    for sentence in doc:
        verb_pps = {}
        for chunk_srcs in sentence:
            chunk_dst = sentence[chunk_srcs.dst]
            if pos_in_phrase('助詞', chunk_srcs) \
                    and pos_in_phrase('動詞', chunk_dst):
                verb_base = obtain_word('動詞', chunk_dst)
                pp_surface = obtain_word('助詞', chunk_srcs)
                phrase = extract_phrase(chunk_srcs)
                pp_phrase = f'{pp_surface}_{phrase}'
                dst_verb = f'{chunk_srcs.dst}_{verb_base}'
                if dst_verb in verb_pps:
                    verb_pps[dst_verb][0].append(pp_surface)
                    verb_pps[dst_verb][1].append(pp_phrase)
                    verb_pps[dst_verb][0].sort()
                    verb_pps[dst_verb][1].sort()
                else:
                    verb_pps[dst_verb] = [[pp_surface], [pp_phrase]]
        verb_pps_all.append(verb_pps)

    for verb_pps in verb_pps_all:
        for dst_verb in verb_pps.keys():
            verb = dst_verb.split('_')[1]
            phrases = [pp_phrases.split('_')[1] for pp_phrases in verb_pps[dst_verb][1]]
            print('{}\t{}\t{}'.format(verb, ' '.join(verb_pps[dst_verb][0]), ' '.join(phrases)))


