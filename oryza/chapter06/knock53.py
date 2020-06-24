from knock52 import read_data
import joblib

def test(feature):
    model = joblib.load('model.pkl')
    print('Prediction: ' + str(model.predict(feature)))
    print('Prediction Probability: ' + str(model.predict_proba(feature)))

if __name__ == "__main__":
    text = open('test.feature.txt')
    feature, label = read_data(text)
    vectorizer = joblib.load('vectorizer.pkl')
    x_test = vectorizer.transform(feature)
    test(x_test)