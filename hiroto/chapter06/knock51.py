'''
51. 特徴量抽出Permalink
学習データ，検証データ，評価データから特徴量を抽出し，それぞれtrain.feature.txt，valid.feature.txt，
test.feature.txtというファイル名で保存せよ． なお，カテゴリ分類に有用そうな特徴量は各自で自由に設計せよ．
記事の見出しを単語列に変換したものが最低限のベースラインとなるであろう．
'''
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.stem.porter import PorterStemmer
from scipy.sparse import csr_matrix, save_npz, load_npz
features = ['CATEGORY', 'TITLE']
cwd = os.getcwd()
train_fname = f"{cwd}/data/train.txt"
valid_fname = f"{cwd}/data/valid.txt"
test_fname = f"{cwd}/data/test.txt"
porter = PorterStemmer()

#トークン化とstemming
def tokenizer_porter(title):
    text = []
    for word in title.split():
        word = word.strip('\',?.!":')
        word = porter.stem(word)
        text.append(word)
    return text


def main():
    train_df = pd.read_table(train_fname, header=None, names=features)
    valid_df = pd.read_table(valid_fname, header=None, names=features)
    test_df = pd.read_table(test_fname, header=None, names=features)
    #一文をベクトル化
    vectorizer = TfidfVectorizer(tokenizer=tokenizer_porter)
    X_train = vectorizer.fit_transform(train_df['TITLE'].values)
    X_valid = vectorizer.transform(valid_df['TITLE'].values)
    X_test = vectorizer.transform(test_df['TITLE'].values)
    print(type(X_train))
    print(type(X_valid))
    print(type(X_test))

    save_npz("./feature/train.feature.npz", X_train)
    save_npz("./feature/valid.feature.npz", X_valid)
    save_npz("./feature/test.feature.npz", X_test)
    '''outputs
    (10684, 12578)
    (1336, 12578)
    (1336, 12578)
    #次元圧縮
    #一文が 1*300 のベクトル
    svd = TruncatedSVD(n_components=300)
    X_train_svd = svd.fit_transform(X_train)
    X_valid_svd = svd.transform(X_valid)
    X_test_svd = svd.transform(X_test)
    '''
    '''
    np.savetxt('./feature/train.feature.txt', X_train.toarray())
    np.savetxt('./feature/valid.feature.txt', X_valid.toarray())
    np.savetxt('./feature/test.feature.txt', X_test.toarray())
    
    with open('./models/51vectorizer.pickle', mode='wb') as f_vec:
        #, open('./models/51svd.pickle', mode='wb') as f_svd:
        #pickle.dump(svd, f_svd)
        pickle.dump(vectorizer, f_vec)
    '''


if __name__ == '__main__':
    main()
