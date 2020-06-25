import pickle
from sklearn.linear_model import LogisticRegression

from knock52 import obtain_data
from knock53 import load_pickle
from knock54 import calculate_accuracy

def print_params(solver, c):
    print('Solver: {}, C: {}'.format(solver, c))

def grid_search(X_train, y_train, X_dev, y_dev, X_test, y_test):
    best_acc = 0
    best_params = {'solver': None, 'c': None}
    best_model = None
    for solver in ['lbfgs', 'sag', 'saga', 'newton-cg']:
        for i in range(9):
            c = 10**(i-4)
            print_params(solver, c)
            print('Solver: {}, C: {}'.format(solver, c))
            lr = LogisticRegression(C=c,
                                    class_weight='balanced',
                                    multi_class='multinomial',
                                    random_state=1,
                                    solver=solver)
            lr.fit(X_train, y_train)
            dev_acc = calculate_accuracy(y_dev, lr.predict(X_dev))
            print('# dev: {}'.format(format(dev_acc, '.3f')))
            if dev_acc > best_acc:
                best_acc = dev_acc
                best_params['solver'] = solver
                best_params['c'] = c
                best_model = lr
    print('best_parameter')
    print('Solver: {}, C: {}'.format(best_params['solver'], best_params['c']))
    return best_model

    
if __name__ == '__main__':
    data_train = './train.feature.txt'
    data_dev = './dev.feature.txt'
    data_test = './test.feature.txt'

    # obtain test data
    X_train, y_train = obtain_data(data_train, labeled=True)
    X_dev, y_dev = obtain_data(data_dev, labeled=True)
    X_test, y_test = obtain_data(data_test, labeled=True)

    # load vectorizer
    vectorizer = load_pickle('./vectorizer.pkl')
    X_train = vectorizer.transform(X_train)
    X_dev = vectorizer.transform(X_dev)
    X_test = vectorizer.transform(X_test)
    
    best_model =  grid_search(X_train, y_train, X_dev, y_dev, X_test, y_test)
    y_hat = best_model.predict(X_test)
    test_acc = calculate_accuracy(y_test, y_hat)
    print('Accuracy: {}'.format(format(test_acc, '.3f')))

