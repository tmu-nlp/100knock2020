import pandas as pd
import json
from sklearn.feature_extraction.text import HashingVectorizer


def write_to_file(array, filename):
    with open(filename, 'w') as file:
        for line in array:
            file.write(','.join(line) + '\n')


def extract_feture(train, val, test, data_dimension=256):
    text = [line for line in train['TITLE']]  # + [line for line in train['STORY']]
    # text = [line for line in train['URL']]
    vectorizer = HashingVectorizer(n_features=data_dimension)
    # vectorizer.fit(text)
    # print(vectorizer.vocabulary_)

    feature_train = pd.DataFrame(vectorizer.transform(train['TITLE']).toarray())
    feature_val = pd.DataFrame(vectorizer.transform(val['TITLE']).toarray())
    feature_test = pd.DataFrame(vectorizer.transform(test['TITLE']).toarray())
    # bow_vocab = pd.DataFrame([[key, value] for (key, value) in vectorizer.vocabulary_.items()])
    # print(train.shape, feature_train.shape)

    feature_train.to_csv('train.feature.txt', header=False, index=False)
    feature_val.to_csv('valid.feature.txt', header=False, index=False)
    feature_test.to_csv('test.feature.txt', header=False, index=False)
    # bow_vocab.to_csv('vocab.bow.txt', header=False, index=False)

    return feature_train, feature_val, feature_test


if __name__ == '__main__':
    train = pd.read_csv('train.txt', sep='\t')
    val = pd.read_csv('valid.txt', sep='\t')
    test = pd.read_csv('test.txt', sep='\t')

    extract_feture(train, val, test)
