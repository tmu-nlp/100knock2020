'''
55. 混同行列の作成Permalink
52で学習したロジスティック回帰モデルの混同行列（confusion matrix）を，学習データおよび評価データ上で作成せよ．
'''
import pickle
import pandas as pd
import numpy as np
from knock51 import tokenizer_porter
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix, save_npz, load_npz
#混同行列をラベル付きで表示してくれるライブラリ
from pycm import ConfusionMatrix
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

#訓練データでの混同行列
labels_train_pred = clf.predict(sc.transform(X_train))
labels_train_true = le.transform(train_df['CATEGORY'].values)
cm_train = ConfusionMatrix(le.inverse_transform(labels_train_true)\
    , le.inverse_transform(labels_train_pred))
print('By training data\n')
cm_train.print_matrix()

#評価データでの混同行列
labels_test_pred = clf.predict(sc.transform(X_test))
labels_test_true = le.transform(test_df['CATEGORY'].values)
cm_test = ConfusionMatrix(le.inverse_transform(labels_test_true)\
    , le.inverse_transform(labels_test_pred))
print('By test data\n')
cm_test.print_matrix()