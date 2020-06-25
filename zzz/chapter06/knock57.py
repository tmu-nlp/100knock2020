from sklearn.externals import joblib
import numpy as np
from zzz.chapter06.knock52 import load_data

if __name__ == '__main__':
    clf = joblib.load('linear_model.pkl')
    print(clf.coef_.shape)
    # [b, e, m, t]
    w_b, w_e, w_m, w_t = clf.coef_[0], clf.coef_[1], clf.coef_[2], clf.coef_[3]
    print('Weights of class b\t', 'max 10:', np.sort(w_b)[:-11:-1], '\tmin 10:', np.sort(w_b)[:10])
    print('Weights of class e\t', 'max 10:', np.sort(w_e)[:-11:-1], '\tmin 10:', np.sort(w_e)[:10])
    print('Weights of class m\t', 'max 10:', np.sort(w_m)[:-11:-1], '\tmin 10:', np.sort(w_m)[:10])
    print('Weights of class t\t', 'max 10:', np.sort(w_t)[:-11:-1], '\tmin 10:', np.sort(w_t)[:10])
