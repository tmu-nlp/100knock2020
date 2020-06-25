from sklearn.externals import joblib

from zzz.chapter06.knock52 import load_data

if __name__ == '__main__':
    clf = joblib.load('linear_model.pkl')
    (x_val, y_val) = load_data('valid{}.txt')
    x = x_val.sample(n=10)
    # [b, e, m, t]
    res = clf.predict(x)
    prob = clf.predict_proba(x)

    print('Category: X', '|\tprobability:', ['b', 'e', 'm', 't'])
    for (r, p) in zip(res, prob):
        print('Category:', r, '|\tprobability:', p)