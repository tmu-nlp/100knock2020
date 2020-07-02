from gensim import models
from math import sqrt
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

w = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
csv = pd.read_csv('wordsim353/combined.csv', sep=',')
word1 = csv['Word 1']
word2 = csv['Word 2']
human_score = csv['Human (mean)']

x = []
y = []
for (w1, w2, s) in zip(word1, word2, human_score):
    y.append(float(s))
    x.append(w.similarity(w1, w2))

rho = spearmanr(x, y)
print(rho[0])

