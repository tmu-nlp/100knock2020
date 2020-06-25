import string
from nltk.corpus import stopwords

def obtain_feature(headline):
    stop_words = set(stopwords.words('english'))
    punctuations = set(string.punctuation)
    hl_lower = [word.lower() for word in headline.split()]
    features = [word for word in hl_lower if word not in stop_words and word not in punctuations]
    return features 

def read_file(file_path):
    cat_features = []
    with open(file_path) as fp:
        for article in fp:
            category, headline = article.split('\t')
            feature = obtain_feature(headline)
            cat_feature = [category, feature]
            cat_features.append(cat_feature)
    return cat_features

def write_features(cat_features, file_name):
    with open(file_name, 'w') as fp:
        for cat_feature in cat_features:
            cat = cat_feature[0]
            feature = cat_feature[1]
            fp.write('{}\t{}\n'.format(cat, ' '.join(feature)))

if __name__ == '__main__':
    train_file = './train.txt'
    dev_file = './dev.txt'
    test_file = './test.txt'

    cat_features = read_file(train_file)
    write_features(cat_features, file_name='./train.feature.txt')

    cat_features = read_file(dev_file)
    write_features(cat_features, file_name='./dev.feature.txt')

    cat_features = read_file(test_file)
    write_features(cat_features, file_name='./test.feature.txt')

