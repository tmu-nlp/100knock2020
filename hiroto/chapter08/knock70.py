import torch
import pickle
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
#ラベルを整数に直す
lable_mapping = {'b':0, 't':1, 'e':2, 'm':3}
#googleのモデル持ってくる
with open('./models/google_model.pickle', mode='rb') as f:
    model = pickle.load(f)

#token化
def tokenize(sentence):
    words = word_tokenize(sentence)
    words = [word for word in words]
    return words

#文をベクトル化
def vectorize(words):
    vectors = []
    T = 0
    for word in words:
        if word in model:
            vectors.append(torch.tensor(model[word]))
            T += 1
        else: pass
    return sum(vectors) / T

#データを読み込み，ベクトル化もする
def read_dataset_and_vectorize(fname):
    labels, vectors = [], []
    with open(fname) as file:
        for line in file:
            cols = line.split('\t')
            labels.append(lable_mapping[cols[0]])
            words = tokenize(cols[1])
            vectors.append(vectorize(words))
    labels = torch.tensor(labels)        
    vectors = torch.stack(vectors)
    return labels, vectors

def main():
    train_labels, train_vectors = read_dataset_and_vectorize('../chapter06/data/train.txt')
    valid_labels, valid_vectors = read_dataset_and_vectorize('../chapter06/data/valid.txt')
    test_labels, test_vectors = read_dataset_and_vectorize('../chapter06/data/test.txt')
    '''
    for i in ['labels', 'vectors']:
        for j in ['train', 'valid', 'test']:
            with open(f'./data/{j}_{i}.pickle') as file:
                pickle.dump()
    '''

    with open('./data/train_labels.pickle', mode='wb') as train_l\
        , open('./data/valid_labels.pickle', mode='wb') as valid_l\
        , open('./data/test_labels.pickle', mode='wb') as test_l\
        , open('./data/train_vectors.pickle', mode='wb') as train_v\
        , open('./data/valid_vectors.pickle', mode = 'wb') as valid_v\
        , open('./data/test_vectors.pickle', mode = 'wb') as test_v:
        pickle.dump(train_labels, train_l)
        pickle.dump(valid_labels, valid_l)
        pickle.dump(test_labels, test_l)
        pickle.dump(train_vectors, train_v)
        pickle.dump(valid_vectors, valid_v)
        pickle.dump(test_vectors, test_v)

if __name__ == '__main__':
    main()