from sklearn.externals import joblib
from sklearn.linear_model import RidgeClassifier
import numpy as np
import matplotlib.pyplot as plt

from zzz.chapter06.knock52 import load_data
from zzz.chapter06.knock56 import generate_p_r_f1

if __name__ == '__main__':
    clf = joblib.load('linear_model.pkl')
    (x_train, y_train) = load_data('train{}.txt')
    (x_val, y_val) = load_data('valid{}.txt')
    (x_test, y_test) = load_data('test{}.txt')

    res_train = []
    res_val = []
    res_test = []
    alpha = np.linspace(0.0, 2.0, 500)

    for a in alpha:
        model = RidgeClassifier(alpha=a)
        model.fit(x_train, y_train)

        *_, f1_train = generate_p_r_f1(model, x_train, y_train)
        *_, f1_val = generate_p_r_f1(model, x_val, y_val)
        *_, f1_test = generate_p_r_f1(model, x_test, y_test)

        print(a, f1_train, f1_val, f1_test)
        res_train.append(f1_train)
        res_val.append(f1_val)
        res_test.append(f1_test)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(alpha, res_train, 'r', label='train')
    ax.plot(alpha, res_val, 'b', label='valid')
    ax.plot(alpha, res_test, 'g', label='test')
    ax.set_xlabel('Regularization Parameter')
    ax.set_ylabel('F1-score macro')
    ax.legend(loc='upper right')
    plt.show()