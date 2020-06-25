from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from zzz.chapter06.knock52 import load_data


def generate_confusion_matrix(model, x, y):
    y_pred = model.predict(x)
    c_matrix = confusion_matrix(y, y_pred)
    return c_matrix


if __name__ == '__main__':
    clf = joblib.load('linear_model.pkl')
    (x_train, y_train) = load_data('train{}.txt')
    (x_val, y_val) = load_data('valid{}.txt')

    train_matrxi = generate_confusion_matrix(clf, x_train, y_train)
    print(train_matrxi)

    val_matrix = generate_confusion_matrix(clf, x_val, y_val)
    print(val_matrix)