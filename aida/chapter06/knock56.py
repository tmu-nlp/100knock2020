import pickle

from knock52 import obtain_data
from knock53 import generate_label_prob, load_pickle
from knock55 import make_confusion_matrix

def print_f1(y_test, y_hat):
    matrix = make_confusion_matrix(y_test, y_hat)
    labels = sorted(list(set(y_test)))
    label_dic = {i:labels[i] for i in range(len(labels))}

    tps = []; p_hats = []; p_anses = []
    precisions = []; recalls = []; f1s = []

    for i in range(len(matrix)):
        tp = matrix[i][i]
        p_hat = sum([matrix[j][i] for j in range(len(matrix))])
        p_ans = sum(matrix[i])
        precision = tp / p_hat
        recall = tp / p_ans
        f1 = 2*recall*precision / (recall + precision)

        tps.append(tp)
        p_hats.append(p_hat); p_anses.append(p_ans)
        precisions.append(precision); recalls.append(recall)
        f1s.append(f1)
    print('Micro') 
    micro_precision = sum(tps) / sum(p_hats)
    micro_recall = sum(tps) / sum(p_anses)
    micro_f1 = 2*micro_precision*micro_recall / (micro_precision + micro_recall)
    print(' Precision: {}\n Recall: {}\n F1: {}'.format(format(micro_precision, '.3f'), format(micro_recall, '.3f'), format(micro_f1, '.3f')))

    print('Macro') 
    macro_precision = sum(precisions) / len(labels)
    macro_recall = sum(recalls) / len(labels)
    macro_f1 = sum(f1s) / len(labels)
    #macro_f1 = 2*macro_precision*macro_recall / (macro_precision + macro_recall)
    print(' Precision: {}\n Recall: {}\n F1: {}'.format(format(macro_precision, '.3f'), format(macro_recall, '.3f'), format(macro_f1, '.3f')))


if __name__ == '__main__':
    data_test = './test.feature.txt'

    # obtain test data
    X_test, y_test = obtain_data(data_test, labeled=True)

    # load vectorizer
    vectorizer = load_pickle('./vectorizer.pkl')

    # load trained model
    lr = load_pickle('./multiclassifier.pkl')

    y_hat, probs = generate_label_prob(lr, vectorizer, X_test)
    print_f1(y_test, y_hat)


