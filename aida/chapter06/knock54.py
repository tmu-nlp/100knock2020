import pickle

from knock52 import obtain_data
from knock53 import generate_label_prob, load_pickle

def calculate_accuracy(y_test, y_hat):
    true_count = sum([y_test[i]==y_hat[i] for i in range(len(y_test))])
    accuracy = true_count / len(y_hat)
    return accuracy


if __name__ == '__main__':
    data_test = './test.feature.txt'

    # obtain test data
    X_test, y_test = obtain_data(data_test, labeled=True)

    # load vectorizer
    vectorizer = load_pickle('./vectorizer.pkl')

    # load trained model
    lr = load_pickle('./multiclassifier.pkl')

    y_hat, probs = generate_label_prob(lr, vectorizer, X_test)

    accuracy = calculate_accuracy(y_test, y_hat)
    print('Accuracy: {}'.format(format(accuracy, '.3f')))

