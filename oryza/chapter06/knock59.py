from knock52 import read_data, vectorize
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

# Search for the training algorithms and 
# parameters that achieves the best accuracy score on the validation data. 
# Then compute its accuracy score on the test data.

if __name__ == "__main__":
    vectorizer = joblib.load('vectorizer.pkl')

    train = open('train.feature.txt')
    train_ftr, train_label = read_data(train)
    valid = open('valid.feature.txt')
    valid_ftr, valid_label = read_data(valid)
    test = open('test.feature.txt')
    test_ftr, test_label = read_data(test)

    x_train = vectorizer.transform(train_ftr)
    y_train = train_label
    x_valid = vectorizer.transform(valid_ftr)
    y_valid = valid_label
    x_test = vectorizer.transform(test_ftr)
    y_test = test_label
    
    # Multinomial NB
    nb_model = MultinomialNB()
    nb_model.fit(x_train,y_train)

    # Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(x_train,y_train)

    # SVM
    svm_model = LinearSVC(multi_class = 'crammer_singer', class_weight='balanced')
    svm_model.fit(x_train,y_train)

    print('\nValid:')
    print(nb_model.score(x_valid,y_valid))
    print(rf_model.score(x_valid,y_valid))
    print(svm_model.score(x_valid,y_valid))

    print('\nTest:')
    print(nb_model.score(x_test, y_test))
    print(rf_model.score(x_test, y_test))
    print(svm_model.score(x_test, y_test))

'''
Valid:
LG 0.813008
NB 0.646341
RF 0.760162
SVM 0.804878

Test:
LG 0.828571
NB 0.604081
RF 0.775510
SVM 0.832653
'''