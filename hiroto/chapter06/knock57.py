'''
57. 特徴量の重みの確認Permalink
52で学習したロジスティック回帰モデルの中で，重みの高い特徴量トップ10と，重みの低い特徴量トップ10を確認せよ．
'''
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
from pprint import pprint
from knock51 import tokenizer_porter
features = ['CATEGORY', 'TITLE']

train_df = pd.read_table('./data/train.txt', header=None, names=features)
valid_df = pd.read_table('./data/valid.txt', header=None, names=features)

clf = pickle.load(open('./models/52lr.pickle', mode='rb'))
le = pickle.load(open('./models/52le.pickle', mode='rb'))
vectorizer = pickle.load(open('./models/51vectorizer.pickle', mode='rb'))

classes = le.inverse_transform(clf.classes_)
print(clf.coef_)
print(clf.coef_.shape)
'''clf.coef_, shape:(4, 12578)
[[-3.16399712e-01  6.48075011e-03  2.79094373e-03 ... -2.61160864e-02
  -2.61183978e-02  2.65594553e-02]
 [ 2.90448583e-01  1.48643365e-02  2.02847870e-02 ...  2.92952097e-02
   8.18220491e-03 -4.31015950e-02]
 [ 1.85527697e-02 -1.00769785e-02 -5.42813456e-03 ...  1.59207638e-03
  -3.04324266e-04 -2.48581800e-02]
 [ 2.10980155e-02 -1.71444469e-02 -9.85707621e-03 ...  3.13102243e-03
   3.11896754e-02  2.39905637e-02]]

'''
feature_names = np.array(vectorizer.get_feature_names())

'''feature_names
['' '#23435' '#4' ... 'â€“' '–' '—']
'''
for label, coef in zip(classes, clf.coef_):
    top10_index = np.argsort(coef)[::-1][0:10]
    bottom10_index = np.argsort(coef)[0:10]
    print(f'{label}: TOP10 features')
    print([feature_names[i] for i in top10_index])
    print(f'{label}: BOTTOM10 features')
    print([feature_names[i] for i in bottom10_index])
    print()

'''
b: TOP10 features
['bank', 'china', 'ukrain', 'obamacar', 'fed', 'fall', 'profit', 'bond', 'updat', 'ecb']
b: BOTTOM10 features
['climat', '', 'earli', '-', 'courtney', 'facebook', 'e-cigarett', 'the', 'googl', 'kardashian']

e: TOP10 features
['-', 'kardashian', 'miley', 'cyru', 'kim', 'wed', '', 'chri', 'bieber', 'her']
e: BOTTOM10 features
['updat', 'googl', 'fine', 'ukrain', 'appl', 'facebook', 'climat', 'mer', 'bank', 'iphon']

m: TOP10 features
['ebola', 'mer', 'e-cigarett', 'fda', "alzheimer'", 'cancer', 'outbreak', 'studi', 'vaccin', 'hiv']
m: BOTTOM10 features
['condit', 'glaxo', 'j&j', 'sharp', '20', 'boston', 'climat', 'give', 'dimon', 'mysteri']

t: TOP10 features
['climat', 'facebook', 'googl', 'microsoft', 'appl', 'fcc', 'iphon', 'neutral', 'heartble', 'comcast']
t: BOTTOM10 features
['respons', 'uphold', 'acquisit', 'revenu', 'compens', 'camaro', '824000', '5.95', 'board', 'origin']
'''