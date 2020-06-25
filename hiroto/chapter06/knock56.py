'''
56. 適合率，再現率，F1スコアの計測Permalink
52で学習したロジスティック回帰モデルの適合率，再現率，F1スコアを，評価データ上で計測せよ．
カテゴリごとに適合率，再現率，F1スコアを求め，カテゴリごとの性能をマイクロ平均（micro-average）と
マクロ平均（macro-average）で統合せよ．
'''
import pickle
import pandas as pd
import numpy as np
from pprint import pprint
from knock51 import tokenizer_porter
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.metrics import classification_report, accuracy_score\
    , recall_score, precision_score, f1_score, precision_recall_fscore_support

features = ['CATEGORY', 'TITLE']

train_df = pd.read_table('./data/train.txt', header=None, names=features)
test_df = pd.read_table('./data/test.txt', header=None, names=features)

clf = pickle.load(open('./models/52lr.pickle', mode='rb'))
vectorizer = pickle.load(open('./models/51vectorizer.pickle', mode='rb'))
le = pickle.load(open('./models/52le.pickle', mode='rb'))
sc = pickle.load(open('./models/52sc.pickle', mode='rb'))

X_test_sparse = load_npz("./feature/test.feature.npz")
X_test = X_test_sparse.toarray()
X_test_std = sc.transform(X_test)



#評価データ
labels_test_pred = le.inverse_transform(clf.predict(X_test_std))
labels_test_true = test_df['CATEGORY'].values

'''prf_supportの中身
            0           1           2           3
(array([0.87288136, 0.87477954, 0.73529412, 0.72972973]), # acc
 array([0.91800357, 0.93233083, 0.56179775, 0.52597403]), # recall
 array([0.89487402, 0.90263876, 0.63694268, 0.61132075]), # precision
 array([561, 532,  89, 154]))                             # support ?
'''
prf_support = precision_recall_fscore_support(labels_test_true, labels_test_pred)
for i, label in enumerate(le.classes_):
    print(f'{label}:')
    for j, score_type in enumerate(['precision', 'recall', 'f1-score']):
        print(f'{score_type}: {prf_support[j][i]}')
    print()

for average_type in ['micro', 'macro']:
    prf_support = precision_recall_fscore_support(labels_test_true, labels_test_pred\
        , average=average_type)
    print(f'{average_type}: ')
    for j, score_type in enumerate(['precision', 'recall', 'f1-score']):
        print(f'{score_type}: {prf_support[j]}')
    print()

print(classification_report(labels_test_true, labels_test_pred))

'''
b:
precision: 0.9045996592844975
recall: 0.946524064171123
f1-score: 0.9250871080139373

e:
precision: 0.9382022471910112
recall: 0.9417293233082706
f1-score: 0.9399624765478424

m:
precision: 0.8461538461538461
recall: 0.7415730337078652
f1-score: 0.7904191616766467

t:
precision: 0.8175182481751825
recall: 0.7272727272727273
f1-score: 0.7697594501718212

micro: 
precision: 0.905688622754491
recall: 0.905688622754491
f1-score: 0.905688622754491

macro: 
precision: 0.8766185002011343
recall: 0.8392747871149966
f1-score: 0.856307049102562

              precision    recall  f1-score   support

           b       0.90      0.95      0.93       561
           e       0.94      0.94      0.94       532
           m       0.85      0.74      0.79        89
           t       0.82      0.73      0.77       154

    accuracy                           0.91      1336
   macro avg       0.88      0.84      0.86      1336
weighted avg       0.90      0.91      0.90      1336
'''