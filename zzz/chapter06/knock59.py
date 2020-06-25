from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
import numpy as np
import pandas as pd
import warnings
from zzz.chapter06.knock56 import generate_p_r_f1
from zzz.chapter06.knock52 import load_data
from zzz.chapter06.knock51 import extract_feture

'''
hyper-parameters:
    data-dimension: from 10 to 250
algorithms:
    linear: ridge classifier
    neighbors: k neighbors classifier
    decision tree: DT classifier
    SVM: SVC
'''


def ridge_classify(x_train, y_train, x_val, y_val, x_test, y_test):
    res_f1_val = 0.0
    res_f1_test = 0.0
    parameters = ''
    alpha = np.linspace(0.0, 2.0, 500)
    for a in alpha:
        model = RidgeClassifier(alpha=a)
        model.fit(x_train, y_train)

        *_, f1_train = generate_p_r_f1(model, x_train, y_train)
        *_, f1_val = generate_p_r_f1(model, x_val, y_val)
        *_, f1_test = generate_p_r_f1(model, x_test, y_test)

        if f1_val > res_f1_val:
            res_f1_val = f1_val
            res_f1_test = f1_test
            parameters = 'alpha: ' + str(a)
    return res_f1_test, parameters


def decision_tree_classify(x_train, y_train, x_val, y_val, x_test, y_test):
    res_f1_val = 0.0
    res_f1_test = 0.0
    parameters = ''
    for criterion in ['gini', 'entropy']:
        for splitter in ['best', 'random']:
            model = tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter)
            model.fit(x_train, y_train)

            *_, f1_train = generate_p_r_f1(model, x_train, y_train)
            *_, f1_val = generate_p_r_f1(model, x_val, y_val)
            *_, f1_test = generate_p_r_f1(model, x_test, y_test)

            if f1_val > res_f1_val:
                res_f1_val = f1_val
                res_f1_test = f1_test
                parameters = 'criterion: ' + criterion + '\tsplitter: ' + splitter
    return res_f1_test, parameters


def k_neighbors_classify(x_train, y_train, x_val, y_val, x_test, y_test):
    res_f1_val = 0.0
    res_f1_test = 0.0
    parameters = ''
    for n_neighbors in range(1, 15):
        for weights in ['uniform', 'distance']:
            for algorithm in ['auto', 'ball_tree', 'kd_tree', 'brute']:
                model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)

                model.fit(x_train, y_train)

                *_, f1_train = generate_p_r_f1(model, x_train, y_train)
                *_, f1_val = generate_p_r_f1(model, x_val, y_val)
                *_, f1_test = generate_p_r_f1(model, x_test, y_test)

                if f1_val > res_f1_val:
                    res_f1_val = f1_val
                    res_f1_test = f1_test
                    parameters = 'n_neighbors: ' + str(
                        n_neighbors) + '\tweight: ' + weights + '\talgorithm: ' + algorithm
    return res_f1_test, parameters


def support_vector_classify(x_train, y_train, x_val, y_val, x_test, y_test):
    res_f1_val = 0.0
    res_f1_test = 0.0
    parameters = ''
    for C in np.linspace(0.01, 2.0, 20):
        for kernel in ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']:
            model = SVC(C=C, kernel=kernel)
            model.fit(x_train, y_train)

            *_, f1_train = generate_p_r_f1(model, x_train, y_train)
            *_, f1_val = generate_p_r_f1(model, x_val, y_val)
            *_, f1_test = generate_p_r_f1(model, x_test, y_test)

            if f1_val > res_f1_val:
                res_f1_val = f1_val
                res_f1_test = f1_test
                parameters = 'C: ' + str(C) + '\tkernel: ' + kernel
    return res_f1_test, parameters


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    train = pd.read_csv('train.txt', sep='\t')
    val = pd.read_csv('valid.txt', sep='\t')
    test = pd.read_csv('test.txt', sep='\t')

    extract_feture(train, val, test)
    classifiers = [ridge_classify, decision_tree_classify, k_neighbors_classify]
    # classifiers = [support_vector_classify]
    res_f1 = 0.0

    for data_dimension in range(10, 260, 10):
        extract_feture(train, val, test, data_dimension)
        (x_train, y_train) = load_data('train{}.txt')
        (x_val, y_val) = load_data('valid{}.txt')
        (x_test, y_test) = load_data('test{}.txt')

        for classifier in classifiers:
            f1, parameters = classifier(x_train, y_train, x_val, y_val, x_test, y_test)
            if f1 > res_f1:
                res_f1 = f1
                print('=' * 20)
                print('Classifier:', classifier.__name__)
                print('Data dimension:', str(data_dimension))
                print('F1-Score:', res_f1)
                print('Parameters:', parameters)
