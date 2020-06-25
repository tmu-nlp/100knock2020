# 別ファイルのプログラムをインポート
from chapter06 import knock50
from chapter06 import knock51
from chapter06 import knock52
from chapter06 import knock54

import itertools

def calc_scores(C, solver, class_weight):
    train_correct = knock50.train_df['CATEGORY']
    valid_correct = knock50.valid_df['CATEGORY']
    test_correct = knock50.test_df['CATEGORY']

    logistic = knock52.LogisticRegression(C=C, solver=solver, class_weight=class_weight)
    logistic.fit(knock51.train_value, train_correct)

    train_pred_correct = logistic.predict(knock51.train_value)
    valid_pred_correct = logistic.predict(knock51.valid_value)
    test_pred_correct = logistic.predict(knock51.test_value)

    scores = []
    scores.append(knock54.accuracy_score(train_correct, train_pred_correct))
    scores.append(knock54.accuracy_score(valid_correct, valid_pred_correct))
    scores.append(knock54.accuracy_score(test_correct, test_pred_correct))
    return scores

C = knock51.np.logspace(-5, 4, 10, base=10)
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
class_weight = [None, 'balanced']
best_parameter = None
best_scores = None
max_valid_score = 0

for c, s, w in itertools.product(C, solver, class_weight):
    print(c, s, w)
    scores = calc_scores(c, s, w)
    # print (scores)
    if scores[1] > max_valid_score:
        max_valid_score = scores[1]
        best_parameter = [c, s, w]
        best_scores = scores

print('best patameter: ', best_parameter)
print('best scores: ', best_scores)
print('test accuracy: ', best_scores[2])
