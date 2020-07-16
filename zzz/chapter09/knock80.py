import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

PATH = '/Users/zz_zhang/勉強会/100本ノック/100knock2020/zzz/chapter06/'


def text_encode(train, val, test, type='onehot', maxlen=20):
    label_dict = {'b': 0, 't': 1, 'e': 2, 'm': 3}
    train_label = train['CATEGORY']
    val_label = val['CATEGORY']
    test_label = test['CATEGORY']
    for (key, value) in label_dict.items():
        train_label = train_label.replace(key, value)
        val_label = val_label.replace(key, value)
        test_label = test_label.replace(key, value)
    train_label = to_categorical(train_label, num_classes=4)
    val_label = to_categorical(val_label, num_classes=4)
    test_label = to_categorical(test_label, num_classes=4)

    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
    tokenizer.fit_on_texts(train['TITLE'])
    word_index = tokenizer.word_index
    vocab = tokenizer.word_index
    train_id = tokenizer.texts_to_sequences(train['TITLE'])
    val_id = tokenizer.texts_to_sequences(val['TITLE'])
    test_id = tokenizer.texts_to_sequences(test['TITLE'])
    if type == 'seq':
        train_id = pad_sequences(train_id, padding='post', maxlen=maxlen)
        val_id = pad_sequences(val_id, padding='post', maxlen=maxlen)
        test_id = pad_sequences(test_id, padding='post', maxlen=maxlen)
        return train_id, train_label, val_id, val_label, test_id, test_label, vocab, word_index
    else:
        train_onehot = tokenizer.sequences_to_matrix(train_id, mode='binary')
        val_onehot = tokenizer.sequences_to_matrix(val_id, mode='binary')
        test_onehot = tokenizer.sequences_to_matrix(test_id, mode='binary')
        return train_onehot, train_label, val_onehot, val_label, test_onehot, test_label, vocab, word_index


if __name__ == '__main__':
    train = pd.read_csv(PATH + 'train.txt', sep='\t')
    val = pd.read_csv(PATH + 'valid.txt', sep='\t')
    test = pd.read_csv(PATH + 'test.txt', sep='\t')

    train_id, train_label, val_id, val_label, test_id, test_label, vocab, word_index = text_encode(train, val, test, type='seq')

    print(train_id[:5])
    print(train_label[:5])
