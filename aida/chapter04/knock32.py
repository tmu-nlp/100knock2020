from knock30 import read_file

doc = read_file()
verb_surfaces = []
for morphs in doc:
    for morph in morphs:
        if morph['pos'] == '動詞':
            verb_surfaces.append(morph['surface'])

for verb in verb_surfaces:
    print(verb)

