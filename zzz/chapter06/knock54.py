from sklearn.externals import joblib
from zzz.chapter06.knock52 import load_data

if __name__ == '__main__':
    clf = joblib.load('linear_model.pkl')
    (x_train, y_train) = load_data('train{}.txt')
    (x_val, y_val) = load_data('valid{}.txt')

    print('Accuracy in train set:', clf.score(x_train, y_train))
    print('Accuracy in validation set:',clf.score(x_val, y_val))