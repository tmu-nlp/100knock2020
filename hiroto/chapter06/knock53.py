'''
53. 予測Permalink
52で学習したロジスティック回帰モデルを用い，与えられた記事見出しからカテゴリと
その予測確率を計算するプログラムを実装せよ．
'''
import pickle
import pandas as pd
from knock51 import tokenizer_porter
features = ['CATEGORY', 'TITLE']

train_df = pd.read_table('./data/train.txt', header=None, names=features)

clf = pickle.load(open('./models/52lr.pickle', mode='rb'))
vectorizer = pickle.load(open('./models/51vectorizer.pickle', mode='rb'))
le = pickle.load(open('./models/52le.pickle', mode='rb'))
sc = pickle.load(open('./models/52sc.pickle', mode='rb'))

def predict_proba(title):
    title = [title]
    title = vectorizer.transform(title)
    title_std = sc.transform(title.toarray())
    probas_list = clf.predict_proba(title_std)
    labels = le.inverse_transform(clf.classes_)
    for probas in probas_list:
        for label, proba in zip(labels, probas):
            print(f'{label}: {proba}')

#trainの最初の５行を予測
for row in range(5):
    text = train_df.iloc[row]['TITLE']
    print(text)
    predict_proba(text)

'''
Bulgaria's third biggest lender says no restrictions on operations
b: 0.9982125997073096
e: 2.3159194939373254e-05
m: 0.0007542378642938013
t: 0.0010100032334572774
Party tents go up on the eve of Jessica Simpson's wedding
b: 0.0002546948620452731
e: 0.9985183732829057
m: 0.0007026252212006814
t: 0.0005243066338482575
UK shares slide on China growth concerns, geopolitical tension
b: 0.9996002157765644
e: 0.00010976225961677674
m: 0.00023084145157303498
t: 5.9180512245807525e-05
Scott Derrickson To Direct "Dr. Strange" - Are We In For A Darker Marvel Flick?
b: 5.335144145804637e-05
e: 0.9987738941726373
m: 0.00048043636907535285
t: 0.0006923180168293036
Obama dedicates National September 11 Memorial Museum as 'a sacred place  ...
b: 0.00022610924133175737
e: 0.9984299634812612
m: 0.0007841934278720919
t: 0.0005597338495350208
'''