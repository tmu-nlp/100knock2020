import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


def read_feature(filename, vocab_filename=''):
    data = pd.read_csv(filename, header=None)
    if len(vocab_filename) > 0:
        vocab = pd.read_csv(vocab_filename)
        return data, vocab
    return data


def read_label(filename):
    label = pd.read_csv(filename, sep='\t')
    label = label['CATEGORY']
    return label


def load_data(filename):
    data = pd.read_csv(filename.format('.feature'), header=None)
    label = pd.read_csv(filename.format(''), sep='\t')
    label = label['CATEGORY']
    return (data, label)


if __name__ == '__main__':
    clf = LogisticRegression()
    (x_train, y_train) = load_data('train{}.txt')
    (x_val, y_val) = load_data('valid{}.txt')
    # print(x_train[:10])
    # print(y_train)
    clf.fit(x_train, y_train)
    print(clf.score(x_train, y_train))
    print(clf.score(x_val, y_val))
    joblib.dump(clf, 'linear_model.pkl')
