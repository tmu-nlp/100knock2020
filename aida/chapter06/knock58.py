import pickle
from sklearn.linear_model import LogisticRegression

from knock52 import obtain_data
from knock53 import load_pickle
from knock54 import calculate_accuracy

def change_norm(X_train, y_train, X_dev, y_dev, X_test, y_test):
    test_accs = []
    for param in range(9):
        c = 10**(param-4)
        lr = LogisticRegression(C=c,
                                class_weight='balanced',
                                multi_class='multinomial',
                                random_state=1,
                                solver='lbfgs')
        lr.fit(X_train, y_train)
        print('Normalize parameter: {}'.format(c))
        train_acc = calculate_accuracy(y_train, lr.predict(X_train))
        print('# train: {}'.format(format(train_acc, '.3f')))
        dev_acc = calculate_accuracy(y_dev, lr.predict(X_dev))
        print('# dev: {}'.format(format(dev_acc, '.3f')))
        test_acc = calculate_accuracy(y_test, lr.predict(X_test))
        print('# test: {}'.format(format(test_acc, '.3f')))
        test_accs.append(test_acc)
    return test_accs

def plot_accs(test_accs):
    import matplotlib.pyplot as plt
    plt.xlabel('Normalize parameter: C')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    cs = [10**(i-4) for i in range(9)]
    plt.plot(cs, test_accs)
    plt.savefig('NormC_Accuracy.png')
    
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

    test_accs = change_norm(X_train, y_train, X_dev, y_dev, X_test, y_test)
    plot_accs(test_accs)


