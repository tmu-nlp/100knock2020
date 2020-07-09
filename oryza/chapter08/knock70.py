from gensim.models import KeyedVectors
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib

def get_mean_vector(word2vec_model, words):
    # remove out-of-vocabulary words
    words = [word for word in words if word in word2vec_model.vocab]
    if len(words) >= 1:
        return np.mean(word2vec_model[words], axis=0)
    else:
        return []

def get_doc_vector(corpus, model, ftr_dump, label_dump):
    corpus = corpus.readlines()
    labelencoder = LabelEncoder()
    labels = []
    features = np.empty((0,300))
    for doc in corpus:
        line = doc.strip().split('\t')
        feature_vec = get_mean_vector(model, line[0])
        labels.append(line[1])
        if len(feature_vec) > 0:
            features = np.append(features, [feature_vec], axis=0)
    
    label_encoded = labelencoder.fit_transform(labels)
    joblib.dump(label_encoded, label_dump)
    joblib.dump(features, ftr_dump)

if __name__ == "__main__":
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    train_file = open('train2.feature.txt')
    get_doc_vector(train_file, model, 'train_feature.pkl', 'train_label.pkl')
    valid_file = open('valid2.feature.txt')
    get_doc_vector(valid_file, model, 'valid_feature.pkl', 'valid_label.pkl')
    test_file = open('test2.feature.txt')
    get_doc_vector(test_file, model, 'test_feature.pkl', 'test_label.pkl')

