from knock52 import read_data
from sklearn import metrics
import joblib

def accuracy(feature, label, model_name, vectorizer_name):
    model = joblib.load(model_name)
    vectorizer = joblib.load(vectorizer_name)
    x = vectorizer.transform(feature)
    prediction = model.predict(x)
    return metrics.accuracy_score(label, prediction)
    
if __name__ == "__main__":
    train = open('train.feature.txt')
    train_ftr, train_label = read_data(train)
    print('Train Accuracy: ' + str(round(accuracy(train_ftr, train_label, 'model.pkl','vectorizer.pkl'),6)))
    
    test = open('test.feature.txt')
    test_ftr, test_label = read_data(test)
    print('Test Accuracy: ' + str(round(accuracy(test_ftr, test_label, 'model.pkl','vectorizer.pkl'),6)))
    
'''
Train Accuracy: 0.996441
Test Accuracy: 0.828571
'''