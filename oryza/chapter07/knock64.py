from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
analogy_file = open('questions-words.txt')

words_sim = []
for line in analogy_file:
    words = line.strip().split(' ')
    if words[0] != ':':
        sim = model.most_similar(positive = [words[1], words[2]], negative = [words[0]], topn = 1)

        for i in sim:
            words_sim.append(words[0] + '\t' + words[1] + '\t' + words[2] + '\t' + str(i[0]) + '\t' + str(i[1]) + '\n')

with open('questions-words-ans.txt','w') as fout:
    for line in words_sim:
        fout.write(line)