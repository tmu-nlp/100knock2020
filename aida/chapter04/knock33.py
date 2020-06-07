from knock30 import read_file

doc = read_file()
phrases = []
for morphs in doc:
    for i in range(1, len(morphs)-1):
        morph = morphs[i]
        if morph['surface'] == 'の':
            context_morph_1 = morphs[i-1]
            context_morph_2 = morphs[i+1]
            if context_morph_1['pos'] == context_morph_2['pos'] == '名詞':
                phrase = context_morph_1['surface'] + morph['surface'] + context_morph_2['surface']
                phrases.append(phrase)

for phrase in sorted(phrases):
    print(phrase)

