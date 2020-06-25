from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def train(feature, label, model_name, c):
    lg = LogisticRegression(
        multi_class = 'multinomial',
        class_weight = 'balanced',
        C = 10**c,
        max_iter = 500
    )
    lg.fit(feature,label)
    joblib.dump(lg, model_name)

def read_data(data):
    text = data.readlines()
    feature = []
    label = []
    for line in text:
        line = line.strip().split('\t')
        feature.append(line[0])
        label.append(line[1])
    return feature, label    

def vectorize(ftr):
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    vectorizer.fit(ftr)
    return vectorizer

if __name__ == "__main__":
    text = open('train.feature.txt')
    feature, label = read_data(text)
    vectorizer = vectorize(feature)
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    x_train = vectorizer.transform(feature)
    y_train = label
    train(x_train, y_train, 'model.pkl', 0)



