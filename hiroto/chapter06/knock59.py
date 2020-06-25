#https://hidehiroqt.com/archives/560
'''
59. ハイパーパラメータの探索Permalink
学習アルゴリズムや学習パラメータを変えながら，カテゴリ分類モデルを学習せよ．
検証データ上の正解率が最も高くなる学習アルゴリズム・パラメータを求めよ．
また，その学習アルゴリズム・パラメータを用いたときの評価データ上の正解率を求めよ．
'''
'''
データセットに対して

liblinear：小さいデータセットに対していい選択
sag、saga：大きいデータセットに対し収束が早い
多次元問題に対して

newton-cg、sag、saga、lbfgs：多項式損失関数を扱える
liblinear：1対他に限られる
正則化に対して

lbfgs、sagはL2正則化に対してのみ使用可能。他は両方可能
'''
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
features = ['CATEGORY', 'TITLE']
#データ取得
train_df = pd.read_table('./data/train.txt', header=None, names=features)
valid_df = pd.read_table('./data/valid.txt', header=None, names=features)
test_df = pd.read_table('./data/test.txt', header=None, names=features)
#特徴量取得
X_train_sparse = load_npz("./feature/train.feature.npz")
X_valid_sparse = load_npz("./feature/valid.feature.npz")
X_test_sparse = load_npz("./feature/test.feature.npz")
X_train, X_valid, X_test = X_train_sparse.toarray(), \
    X_valid_sparse.toarray(), X_test_sparse.toarray()
#モデル取得
#sc => StandardScaler
#le => LabelEncoder
sc = pickle.load(open('./models/52sc.pickle', mode='rb'))
le = pickle.load(open('./models/52le.pickle', mode='rb'))
#スケーリング
X_train_std = sc.transform(X_train)
X_valid_std = sc.transform(X_valid)
X_test_std = sc.transform(X_test)
del X_train, X_valid, X_test
#正解ラベル
y_train = le.transform(train_df['CATEGORY'].values)
y_valid = le.transform(valid_df['CATEGORY'].values)
y_test = le.transform(test_df['CATEGORY'].values)
#最も高い正解率を保存する変数
best_acc = 0
best_clf = None
#正解率が今までより一番高いなら更新し，モデルも保存する関数
def judge_best_model(clf):
    clf.fit(X_train_std, y_train)
    labels_valid_pred = clf.predict(X_valid_std)
    acc = accuracy_score(y_valid, labels_valid_pred)
    global best_acc
    if acc > best_acc:
        best_acc = acc
        global best_clf
        best_clf = clf
    else: pass
    return

#ロジスティック回帰
#solverはliblinearに固定する
#param_rangeはCを指す
param_range = [0.01, 0.1, 1.0]
penalties = ['l1', 'l2']
print('Logistic Regression')
for penalty in tqdm(penalties, desc='penalty'):
    for C in tqdm(param_range, desc='C'):
        lr = LogisticRegression(penalty=penalty, C=C, solver='liblinear'\
            , multi_class='auto', random_state=0)
        judge_best_model(lr)
        del lr

'''
#SVM
#Cとgammaはロジスティック回帰の時と同じものを使用する

kernels = ['linear', 'rbf']
print('SVM')
for kernel in tqdm(kernels, desc='kernel'):
    if kernel == 'rbf':
        for gamma in tqdm(param_range, desc='gamma'):
            for C in tqdm(param_range, desc='C'):
                svc = SVC(C=C, kernel=kernel, gamma=gamma, random_state=0)
                judge_best_model(svc)
    else:
        for C in tqdm(param_range, desc='C'):
            svc = SVC(C=C, kernel=kernel, random_state=0)
            judge_best_model(svc)
'''
#KNN
k_s = list(range(1, 11))
print('KNN')
for k in tqdm(k_s, desc='k'):
    knn = KNeighborsClassifier(n_neighbors=k)
    judge_best_model(knn)
    del knn

#決定木
#不純度，誤差の指標
criterions = ['gini', 'entropy']
#Noneから7までの深さ
max_depths = [None] + list(range(1, 8))
print('Decision Tree')
for criterion in tqdm(criterions, desc='criteion'):
    for max_depth in tqdm(max_depths, desc='max_depth'):
        tree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth\
            , random_state=0)
        judge_best_model(tree)
        del tree


#一番良いモデルのパラメータを表示
print(f'Validation Accuracy: {best_acc}')
print(best_clf.get_params)
#一番良いモデルでテストデータの正解率を計算する
labels_test_pred = best_clf.predict(X_test_std)
print(f'Test Accuracy: {accuracy_score(labels_test_pred, y_test)}')

'''
Validation Accuracy: 0.9139221556886228
<bound method BaseEstimator.get_params of LogisticRegression(C=0.01, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)>
Test Accuracy: 0.9079341317365269
'''