'''
52. 学習Permalink
51で構築した学習データを用いて，ロジスティック回帰モデルを学習せよ
'''
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.sparse import csr_matrix, save_npz, load_npz
features = ['CATEGORY', 'TITLE']

train_df = pd.read_table('./data/train.txt', header=None, names=features)
'''
X_train = np.loadtxt('./feature/train.feature.txt')
X_valid = np.loadtxt('./feature/valid.feature.txt')
X_test = np.loadtxt('./feature/test.feature.txt')
'''
X_train_sparse = load_npz("./feature/train.feature.npz")
X_valid_sparse = load_npz("./feature/valid.feature.npz")
X_test_sparse = load_npz("./feature/test.feature.npz")
X_train, X_valid, X_test = X_train_sparse.toarray(), \
    X_valid_sparse.toarray(), X_test_sparse.toarray()

#標準化
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_valid_std = sc.transform(X_valid)
X_test_std = sc.transform(X_test)

#カテゴリ(アルファベット)を整数に変換
le = LabelEncoder()
labels = le.fit_transform(train_df['CATEGORY'].values)

#ロジスティック回帰
lr = LogisticRegression(solver='liblinear', multi_class='auto', random_state=0)
lr.fit(X_train_std, labels)

with open('./models/52lr.pickle', mode='wb') as f_lr\
    , open('./models/52le.pickle', mode='wb') as f_le\
    , open('./models/52sc.pickle', mode='wb') as f_sc:
    pickle.dump(lr, f_lr)
    pickle.dump(le, f_le)
    pickle.dump(sc, f_sc)