from knock30 import read_file

doc = read_file()
verb_bases = []
for morphs in doc:
    for morph in morphs:
        if morph['pos'] == '動詞':
            verb_bases.append(morph['base'])

for verb in verb_bases:
    print(verb)

