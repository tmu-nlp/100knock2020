from gensim.models import KeyedVectors
import torch

def load_model():
    model = KeyedVectors.load_word2vec_format('./../chapter07/GoogleNews-vectors-negative300.bin.gz', binary=True)
    return model

def obtain_matrixes(file_path):
    with open(file_path) as fp:
        lines = fp.readlines()
    label_dic = {'b': 0, 't': 1, 'e': 2, 'm': 3}
    N = len(lines)                                                                                                            
    X = []
    Y = []
    for i, line in enumerate(lines):
        label, sentence = line.strip().split('\t')
        features = sentence.split()
        common_features = [feature for feature in features if feature in model.wv.vocab]
        if len(common_features)==0:
            print(f'No features line-{i}')
            continue
        x = sum([model[feature] for feature in common_features]) / len(common_features)
        x = torch.from_numpy(x)
        y = torch.Tensor([label_dic[label]])
        Y.append(y)
        X.append(x)
    
    X = torch.stack(X, 0)
    Y = torch.stack(Y, 0)
    return X, Y

if __name__ == '__main__':
    model = load_model()

    train_file = './../chapter06/train.feature.txt'
    dev_file = './../chapter06/dev.feature.txt'
    test_file = './../chapter06/test.feature.txt'

    X_train, Y_train = obtain_matrixes(train_file, model)
    X_dev, Y_dev = obtain_matrixes(dev_file, model)
    X_test, Y_test = obtain_matrixes(test_file, model)

    torch.save(X_train, './tensors/X_train')
    torch.save(Y_train, './tensors/Y_train')
    torch.save(X_dev, './tensors/X_dev')
    torch.save(Y_dev, './tensors/Y_dev')
    torch.save(X_test, './tensors/X_test')
    torch.save(Y_test, './tensors/Y_test')

