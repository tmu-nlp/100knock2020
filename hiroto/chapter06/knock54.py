'''
54. 正解率の計測Permalink
52で学習したロジスティック回帰モデルの正解率を，学習データおよび評価データ上で計測せよ．
'''
import pickle
import numpy as np
import pandas as pd
from knock51 import tokenizer_porter
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix, save_npz, load_npz
features = ['CATEGORY', 'TITLE']

train_df = pd.read_table('./data/train.txt', header=None, names=features)
test_df = pd.read_table('./data/test.txt', header=None, names=features)

clf = pickle.load(open('./models/52lr.pickle', mode='rb'))
vectorizer = pickle.load(open('./models/51vectorizer.pickle', mode='rb'))
le = pickle.load(open('./models/52le.pickle', mode='rb'))
sc = pickle.load(open('./models/52sc.pickle', mode='rb'))

X_train_sparse = load_npz("./feature/train.feature.npz")
X_valid_sparse = load_npz("./feature/valid.feature.npz")
X_test_sparse = load_npz("./feature/test.feature.npz")
X_train, X_valid, X_test = X_train_sparse.toarray(), \
    X_valid_sparse.toarray(), X_test_sparse.toarray()

X_train_std = sc.transform(X_train)
X_valid_std = sc.transform(X_valid)
X_test_std = sc.transform(X_test)


labels_train_pred = clf.predict(X_train_std)
labels_train_true = le.transform(train_df['CATEGORY'].values)
accuracy_train = accuracy_score(labels_train_true, labels_train_pred)

labels_test_pred = clf.predict(X_test_std)
labels_test_true = le.transform(test_df['CATEGORY'].values)
accuracy_test = accuracy_score(labels_test_true, labels_test_pred)

print(f'accuracy_score train: {accuracy_train}')
print(f'accuracy_score test: {accuracy_test}')

'''
accuracy_score train: 0.9997192062897791
accuracy_score test: 0.905688622754491
'''