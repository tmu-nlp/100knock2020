from gensim import models

w = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

with open('questions-words.txt') as file:
    with open('questions-words-ans.txt', 'w') as output:
        for (index, line) in enumerate(file):
            if len(line.split(' ')) != 4:
                output.write(line)
                continue
            w1, w2, w3, w4 = line.split(' ')
            w4 = w4.replace('\n', '')
            # print(v1, v2, v3, v4)
            new_vec = w[w2] - w[w1] + w[w3]
            (w5, sim) = w.similar_by_vector(new_vec, topn=1)[0]

            output.write(w1 + ' ' + w2 + ' ' + w3 + ' ' + w4 + '\t' + w5 + '\n')
            print(w1, w2, w3, w4, '|', w5, sim)