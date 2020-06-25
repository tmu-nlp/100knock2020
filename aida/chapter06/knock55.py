import pickle

from knock52 import obtain_data
from knock53 import generate_label_prob, load_pickle

def make_confusion_matrix(y_test, y_hat):
    # initialize
    cat_num = len(set(y_test))
    matrix = [[0]*cat_num for _ in range(cat_num)]
    labels = sorted(list(set(y_test)))
    label_dic = {labels[i]:i for i in range(len(labels))}

    # calcurate
    for i in range(len(y_test)):
        labelid_ans = label_dic[y_test[i]]
        labelid_hat = label_dic[y_hat[i]]
        matrix[labelid_ans][labelid_hat] += 1
    return matrix


if __name__ == '__main__':
    data_test = './test.feature.txt'

    # obtain test data
    X_test, y_test = obtain_data(data_test, labeled=True)

    # load vectorizer
    vectorizer = load_pickle('./vectorizer.pkl')

    # load trained model
    lr = load_pickle('./multiclassifier.pkl')

    y_hat, probs = generate_label_prob(lr, vectorizer, X_test)
    matrix = make_confusion_matrix(y_test, y_hat)
    print('column: predicted, row: answer')
    for i in range(cat_num):
        for j in range(cat_num):
            print(matrix[i][j], end='\t')
        print()

