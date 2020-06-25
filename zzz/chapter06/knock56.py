from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support
from zzz.chapter06.knock52 import load_data


def generate_p_r_f1(model, x, y, average='macro'):
    y_pred = model.predict(x)
    p, r, f, _ = precision_recall_fscore_support(y, y_pred, beta=1.0, average=average)
    return p, r, f


if __name__ == '__main__':
    clf = joblib.load('linear_model.pkl')
    (x_train, y_train) = load_data('train{}.txt')
    (x_val, y_val) = load_data('valid{}.txt')

    p_train, r_train, f1_train = generate_p_r_f1(clf, x_train, y_train, 'micro')
    print('micro:', p_train, r_train, f1_train)

    p_train, r_train, f1_train = generate_p_r_f1(clf, x_train, y_train, 'macro')
    print('macro:', p_train, r_train, f1_train)
