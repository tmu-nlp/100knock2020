import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def obtain_data(file_path, labeled=True):
    X = []
    y = []
    with open(file_path) as fp:
        for line in fp:
            if labeled:
                ans, sentence = line.strip().split('\t')
            else:
                sentence = line.strip()
                ans = None
            y.append(ans)
            X.append(sentence)
    return X, y

if __name__ == '__main__':
    data_train = './train.feature.txt'

    # obtain training data
    X_train, y_train = obtain_data(data_train, labeled=True)

    # vectorizer
    vectorizer = TfidfVectorizer()
    vectorizer.fit(X_train)
    features = vectorizer.transform(X_train)

    # data into vector
    lr = LogisticRegression(multi_class='multinomial', class_weight='balanced', solver='lbfgs', random_state=1)
    lr.fit(features, y_train)

    # save models
    fp = open('./vectorizer.pkl', 'wb')
    pickle.dump(vectorizer, fp)
    fp = open('./multiclassifier.pkl', 'wb')
    pickle.dump(lr, fp)

