from pprint import pprint

from knock40 import Morph
from knock41 import Chunk, read_file
from knock43 import pos_in_phrase

# extract most left verb
# extract all postpositional particles
def obtain_word(pos, chunk):
    for morph in chunk.morphs:
        if morph.pos == pos:
            if pos == '動詞':
                return morph.base
            if pos == '助詞':
                return morph.surface

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
                dst_verb = f'{chunk_srcs.dst}_{verb_base}'
                if dst_verb in verb_pps:
                    verb_pps[dst_verb].append(pp_surface)
                    verb_pps[dst_verb].sort()
                else:
                    verb_pps[dst_verb] = [pp_surface]
        verb_pps_all.append(verb_pps)

    for verb_pps in verb_pps_all:
        for dst_verb in verb_pps.keys():
            verb = dst_verb.split('_')[1]
            print('{}\t{}'.format(verb, ' '.join(verb_pps[dst_verb])))

# Unix command
## cat output45.txt | sort | uniq -c | sort -n -r | head
"""
  49 する	を
  18 する	が
  17 する	て
  16 する	に
  15 する	と
  11 する	は を
  11 する	に を
   9 する	で を
   9 よる	に
   8 する	が に
"""
## cat output45.txt | grep '^行う\s'| sort | uniq -c | sort -n -r | head
"""
   8 行う	を
   1 行う	まで を
   1 行う	から
   1 行う	に まで を
   1 行う	は を をめぐって
   1 行う	に に により を
   1 行う	て に は は
   1 行う	が て で に
   1 行う	に を を
   1 行う	で に を
"""
## cat output45.txt | grep '^なる\s'| sort | uniq -c | sort -n -r | head
"""
   3 なる	に は
   3 なる	が と
   3 なる	と
   2 なる	に
   1 なる	から が て で と
   1 なる	から で と
   1 なる	て として に は
   1 なる	が と にとって
   1 なる	で と など
   1 なる	が で と に は は
"""
## cat output45.txt | grep '^与える\s'| sort | uniq -c | sort -n -r | head
"""
   1 与える	が など
   1 与える	に は を
   1 与える	が に
"""

