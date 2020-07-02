import numpy as np

with open('./my-answer.txt') as fp:
    for line in fp:
        words = line.strip().split()
        if len(words)==2:
            if 'gram' in words[1]:
                is_syn = True
            else:
                is_syn = False
        else:
            if is_syn:
                syntactic_analogy.append(words[3:5])
            else:
                semantic_analogy.append(words[3:5])

acc_sem = np.mean([ans==pred for ans, pred in semantic_analogy])
acc_syn = np.mean([ans==pred for ans, pred in syntactic_analogy])
print(f'Semantic task, Accuracy: {acc_sem}') 
print(f'Syntactic task, Accuracy: {acc_syn}') 

"""
Semantic task, Accuracy: 0.7308602999210734
Syntactic task, Accuracy: 0.7400468384074942
"""

