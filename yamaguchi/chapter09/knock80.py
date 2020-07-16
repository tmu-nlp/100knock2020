# 数値計算のためのモジュールをインポート
import numpy as np
# データ処理ライブラリをインポート
import pandas as pd
# 単語の出現頻度を数えるためにインポート
from sklearn.feature_extraction.text import CountVectorizer

# knock50で作成した，学習データ，検証データ，評価データを読み込む．
train = pd.read_csv('train.txt', header=None, sep='\t')
valid = pd.read_csv('valid.txt', header=None, sep='\t')
test = pd.read_csv('test.txt', header=None, sep='\t')

# 単語の出現頻度を数える中での最低の値は2
vectorizer = CountVectorizer(min_df=2)
train_title = train.iloc[:,1].str.lower()
cnt = vectorizer.fit_transform(train_title).toarray()
sm = cnt.sum(axis=0)
idx = np.argsort(sm)[::-1]
words = np.array(vectorizer.get_feature_names())[idx]
d = dict()

for i in range(len(words)):
  d[words[i]] = i+1

# 単語にID番号を付与，追加していく．
def get_id(sentence):
  r = []
  for word in sentence:
    r.append(d.get(word,0))
  return r

# 学習データ，検証データ，評価データのそれぞれで出現頻度を計算させる．
def df2id(df):
  ids = []
  for i in df.iloc[:,1].str.lower():
    ids.append(get_id(i.split()))
  return ids

# knock80の実行
X_train = df2id(train)
X_valid = df2id(valid)
X_test = df2id(test)

# 結果をファイルに保存
with open('knock80_train.txt', mode='w', encoding="utf-8") as f:
  print(str(X_train), file=f)
with open('knock80_valid.txt', mode='w', encoding="utf-8") as f:
  print(str(X_valid), file=f)
with open('knock80_test.txt', mode='w', encoding="utf-8") as f:
  print(str(X_test), file=f)

