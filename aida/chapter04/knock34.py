from knock30 import read_file

doc = read_file()
phrases = []
context_count = 0
for morphs in doc:
    for i in range(len(morphs)):
        if context_count > 0:
            context_count -= 1
            continue
        morph = morphs[i]
        if morph['pos'] == '名詞':
            phrase = morph['surface']
            context_count = 0
            for j in range(i+1, len(morphs)):
                context_morph = morphs[j]
                context_count += 1
                if context_morph['pos'] == '名詞':
                    phrase += context_morph['surface']
                else:
                    phrases.append(phrase)
                    break

for phrase in phrases:
    print(phrase)

