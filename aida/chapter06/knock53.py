import pickle

from knock52 import obtain_data

def load_pickle(path):
    fp = open(path, 'rb')
    return pickle.load(fp)

def generate_label_prob(lr, vectorizer, X_test):
    features = vectorizer.transform(X_test)
    y_hat = lr.predict(features)
    probs = lr.predict_proba(features)
    return y_hat, probs


if __name__ == '__main__':
    data_test = './test.feature.txt'

    # obtain test data
    X_test, _ = obtain_data(data_test, labeled=True)

    # load vectorizer
    vectorizer = load_pickle('./vectorizer.pkl')

    # load trained model
    lr = load_pickle('./multiclassifier.pkl')

    y_hat, probs = generate_label_prob(lr, vectorizer, X_test)
    for i in range(len(y_hat)):
        print('{}\t{}'.format(y_hat[i], probs[i]))

