import torch
import pickle

with open('./data/train_labels.pickle', mode='rb') as train_l\
        , open('./data/valid_labels.pickle', mode='rb') as valid_l\
        , open('./data/test_labels.pickle', mode='rb') as test_l\
        , open('./data/train_vectors.pickle', mode='rb') as train_v\
        , open('./data/valid_vectors.pickle', mode = 'rb') as valid_v\
        , open('./data/test_vectors.pickle', mode = 'rb') as test_v:
        train_labels = pickle.load(train_l)
        valid_labels = pickle.load(valid_l)
        test_labels = pickle.load(test_l)
        train_vectors = pickle.load(train_v)
        valid_vectors = pickle.load(valid_v)
        test_vectors = pickle.load(test_v)

#入力は１×３００で出力はカテゴリ数4に応じて４×１のスコア
#重みW
W = torch.randn(300,  4)
scores_v = torch.matmul(train_vectors[0], W)
scores_m = torch.matmul(train_vectors[:4], W)
softmax = torch.nn.Softmax(dim=-1)
y = softmax(scores_v)
Y = softmax(scores_m)
print(y)
print(Y)

'''
tensor([0.0092, 0.3408, 0.0843, 0.5657])
tensor([[0.0092, 0.3408, 0.0843, 0.5657],
        [0.1663, 0.1654, 0.1769, 0.4914],
        [0.0683, 0.4788, 0.0943, 0.3587],
        [0.1771, 0.5756, 0.0181, 0.2291]])
'''