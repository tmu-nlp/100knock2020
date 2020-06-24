import pickle

from knock52 import obtain_data
from knock53 import load_pickle

def print_topk_features(lr, vectorizer, k=10, reverse=True):
    weights = lr.coef_[0].tolist()
    features = vectorizer.get_feature_names()
    weight_features = sorted(zip(weights, features), reverse=reverse)
    if reverse:
        print('-'*5, 'top-k features', '-'*5)
    else:
        print('-'*5, 'bottom-k features', '-'*5)
    for w_f in weight_features[:k]:
        w, f = w_f
        print('{}: {}'.format(f, format(w, '.3f')))
    
if __name__ == '__main__':
    data_test = './test.feature.txt'

    # obtain test data
    X_test, y_test = obtain_data(data_test, labeled=True)

    # load vectorizer
    vectorizer = load_pickle('./vectorizer.pkl')

    # load trained model
    lr = load_pickle('./multiclassifier.pkl')

    # top-k
    print_topk_features(lr, vectorizer, k=10, reverse=True)
    # bottom-k
    print_topk_features(lr, vectorizer, k=10, reverse=False)

