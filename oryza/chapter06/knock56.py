from knock52 import read_data
import joblib
from sklearn import metrics

def calc_score(feature, label):
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    x = vectorizer.transform(feature)
    prediction = model.predict(x)

    print(metrics.classification_report(label, prediction))

if __name__ == "__main__":
    test = open('test.feature.txt')
    test_ftr, test_label = read_data(test)
    calc_score(test_ftr, test_label)