# 別ファイルのプログラムをインポート
from chapter06 import knock50
from chapter06 import knock51
from chapter06 import knock52
from chapter06 import knock54

# グラフを出力するためにインポート
import matplotlib.pyplot as plt

def calc_scores(c):
    train_correct = knock50.train_df['CATEGORY']
    valid_correct = knock50.valid_df['CATEGORY']
    test_correct = knock50.test_df['CATEGORY']

    logistic = knock52.LogisticRegression(C=c)
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
scores = []

for c in C:
    scores.append(calc_scores(c))

scores = knock51.np.array(scores).T
labels = ['train', 'valid', 'test']

for score, label in zip(scores, labels):
    plt.plot(C, score, label=label)

plt.ylim(0, 1.1)
plt.xscale('log')
plt.xlabel('C', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.tick_params(labelsize=14)
plt.grid(True)
plt.legend()
