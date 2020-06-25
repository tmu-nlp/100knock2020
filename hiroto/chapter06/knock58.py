'''
58. 正則化パラメータの変更Permalink
ロジスティック回帰モデルを学習するとき，正則化パラメータを調整することで，学習時の過学習（overfitting）の
度合いを制御できる．異なる正則化パラメータでロジスティック回帰モデルを学習し，学習データ，検証データ，
および評価データ上の正解率を求めよ．実験の結果は，正則化パラメータを横軸，正解率を縦軸としたグラフにまとめよ．
'''
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
features = ['CATEGORY', 'TITLE']

train_df = pd.read_table('./data/train.txt', header=None, names=features)
valid_df = pd.read_table('./data/valid.txt', header=None, names=features)
test_df = pd.read_table('./data/test.txt', header=None, names=features)

X_train_sparse = load_npz("./feature/train.feature.npz")
X_valid_sparse = load_npz("./feature/valid.feature.npz")
X_test_sparse = load_npz("./feature/test.feature.npz")
X_train, X_valid, X_test = X_train_sparse.toarray(), \
    X_valid_sparse.toarray(), X_test_sparse.toarray()

sc = pickle.load(open('./models/52sc.pickle', mode='rb'))

X_train_std = sc.transform(X_train)
X_valid_std = sc.transform(X_valid)
X_test_std = sc.transform(X_test)

del X_train, X_valid, X_test

le = LabelEncoder()
labels = le.fit_transform(train_df['CATEGORY'].values)

#与えられたモデルを用いて，訓練，検証，評価それぞれでの正解率のリストを返す
def cal_accuracy(clf):
    labels_train_pred = clf.predict(X_train_std)
    labels_train_true = le.transform(train_df['CATEGORY'].values)
    accuracy_train = accuracy_score(labels_train_true, labels_train_pred)
    
    labels_valid_pred = clf.predict(X_valid_std)
    labels_valid_true = le.transform(valid_df['CATEGORY'].values)
    accuracy_valid = accuracy_score(labels_valid_true, labels_valid_pred)
    
    labels_test_pred = clf.predict(X_test_std)
    labels_test_true = le.transform(test_df['CATEGORY'].values)
    accuracy_test = accuracy_score(labels_test_true, labels_test_pred)
    
    return [accuracy_train, accuracy_valid, accuracy_test]


params = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
accs_train, accs_valid, accs_test = [], [], []
for param in tqdm(params):
    #Cを大きくしていく
    #Cが小さいほど正則化の度合いが強い
    lr = LogisticRegression(C=param, solver='liblinear', multi_class='auto', random_state=0)
    lr.fit(X_train_std, labels)
    accuracies = cal_accuracy(lr)
    accs_train.append(accuracies[0])
    accs_valid.append(accuracies[1])
    accs_test.append(accuracies[2])

plt.plot(params, accs_train, label='train', marker='.')
plt.plot(params, accs_valid, label='valid', marker='.')
plt.plot(params, accs_test, label='test', marker='.')
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.show()
    