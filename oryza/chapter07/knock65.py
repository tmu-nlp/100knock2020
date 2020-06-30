from gensim.models import KeyedVectors

'''
https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.FastTextKeyedVectors.evaluate_word_analogies
'''
def w2v_model_accuracy(model):
    accuracy = model.accuracy('questions-words-ans2.txt')
    sum_corr = len(accuracy[-1]['correct'])
    sum_incorr = len(accuracy[-1]['incorrect'])
    total = sum_corr + sum_incorr
    percent = lambda a: a / total * 100
    
    print('Total sentences: {}, Correct: {:.2f}%, Incorrect: {:.2f}%'.format(total, percent(sum_corr), percent(sum_incorr)))

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

print(w2v_model_accuracy(model))
print(model.evaluate_word_analogies('questions-words-ans2.txt'))


'''
Total sentences: 12828, Correct: 97.86%, Incorrect: 2.14%
Accuracy 0.9736495388669302
'''