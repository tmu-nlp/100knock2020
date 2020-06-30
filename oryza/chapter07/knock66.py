from gensim.models import KeyedVectors
from scipy import stats

'''
Ref
https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.evaluate_word_pairs
'''

def extract_col(list, col):
    return [float(i.strip().split('\t')[col]) for i in list]

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
# Using evaluate_word_pairs
# print(model.evaluate_word_pairs('combined.tsv'))

# Manual Vector Similarity and Spearman Count
test_file = open('combined.tsv')

human_vector_sim = []
for i in test_file:
    word = i.strip().split('\t')
    similarity = model.similarity(word[0],word[1])
    human_vector_sim.append(word[0] + '\t' + word[1] + '\t' + word[2] + '\t' + str(similarity) + '\n')

print(stats.spearmanr(extract_col(human_vector_sim,2),extract_col(human_vector_sim,3)))

'''
Using evaluate_word_pairs
Return Pearson, Spearman, and oov_ratio 
((0.6238773466616107, 1.7963237724171284e-39), 
SpearmanrResult(correlation=0.6589215888009288, 
pvalue=2.5346056459149263e-45), 0.0)

Manual Vector Similarity and Spearman Count
SpearmanrResult(correlation=0.7000166486272194, 
pvalue=2.86866666051422e-53)
'''