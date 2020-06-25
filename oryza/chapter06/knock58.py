from knock52 import train, read_data, vectorize
from knock54 import accuracy
import matplotlib.pyplot as plt
import joblib

if __name__ == "__main__":
    # load file   
    text = open('train.feature.txt')
    train_ftr, train_label = read_data(text)
    valid = open('valid.feature.txt')
    valid_ftr, valid_label = read_data(valid)
    test = open('test.feature.txt')
    test_ftr, test_label = read_data(test)

    vectorizer = vectorize(train_ftr)
    joblib.dump(vectorizer, 'vectorizer.pkl')

    x_train = vectorizer.transform(train_ftr)
    y_train = train_label
    
    regularization = []
    for c in range(-5,5):
        # train with regularization
        model_name = 'model_reg_10**' + str(c) + '.pkl'
        train(x_train, y_train, model_name, c)
        reg = '10e' + str(c)

        # calculate accuracy on train, valid, test
        acc_train = accuracy(train_ftr, train_label, model_name,'vectorizer.pkl')
        acc_valid = accuracy(valid_ftr, valid_label, model_name,'vectorizer.pkl')
        acc_test= accuracy(test_ftr, test_label, model_name,'vectorizer.pkl')
    
        regularization.append(reg + '\t' + str(round(acc_train,6)) + '\t' + str(round(acc_valid,6)) + '\t' + str(round(acc_test,6)))

    reg_val, acc_tr, acc_val, acc_ts = [],[],[],[]
    for i in regularization:
        temp = i.strip().split('\t')
        reg_val.append(float(temp[0]))
        acc_tr.append(float(temp[1]))
        acc_val.append(float(temp[2]))
        acc_ts.append(float(temp[3]))

    plt.plot(reg_val, acc_tr, color='blue', marker='x', label='train')
    plt.plot(reg_val, acc_val, color='green',  marker='o', label='valid')
    plt.plot(reg_val, acc_ts, color='red',  marker='s', label='test')
    plt.ylabel('Accuracy Score')
    plt.xlabel('Regularization Parameter')
    plt.legend(loc='best')
    plt.xscale('log')
    plt.show()

'''
10**-5  0.964921        0.743902        0.808163
10**-4  0.969497        0.739837        0.812245
10**-3  0.980173        0.780488        0.828571
10**-2  0.980173        0.780488        0.828571
10**-1  0.981698        0.788618        0.82449
10**0   0.996441        0.813008        0.828571
10**1   0.998983        0.821138        0.840816
10**2   0.998983        0.817073        0.840816
10**3   0.998983        0.817073        0.840816
10**4   0.998983        0.817073        0.840816
'''