from knock52 import read_data
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import pandas as pd

def conf_matrix(feature, label):
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    x = vectorizer.transform(feature)
    prediction = model.predict(x)

    cnf_matrix = metrics.confusion_matrix(label, prediction)
    return cnf_matrix
    
    # make confusion matrix with seaborn
    # class_names = ['b','e','m','t'] # name  of classes
    # fig, ax = plt.subplots()
    # tick_marks = np.arange(len(class_names))
    # plt.xticks(tick_marks, class_names)
    # plt.yticks(tick_marks, class_names)
    # # create heatmap
    # sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    # ax.xaxis.set_label_position("top")
    # plt.tight_layout()
    # plt.title('Confusion matrix', y=1.1)
    # plt.ylabel('Actual label')
    # plt.xlabel('Predicted label')
    # plt.show()

if __name__ == "__main__":
    train = open('train.feature.txt')
    train_ftr, train_label = read_data(train)
    print('Train')
    print(conf_matrix(train_ftr, train_label))

    test = open('test.feature.txt')
    test_ftr, test_label = read_data(test)
    print('\nTest')
    print(conf_matrix(test_ftr, test_label))

'''
Train
[[354   0   0   1]
 [  4 983   0   2]
 [  0   0 260   0]
 [  0   0   0 363]]

Test
[[ 36   5   1   3]
 [  3 109   0   3]
 [  1   3  21   2]
 [  8  11   2  37]]
'''  