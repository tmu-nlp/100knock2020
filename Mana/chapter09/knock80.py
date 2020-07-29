# 問題51で構築した学習データ中の単語にユニークなID番号を付与したい．
# 学習データ中で最も頻出する単語に1，2番目に頻出する単語に2，……といった方法で，
# 学習データ中で2回以上出現する単語にID番号を付与せよ．
# そして，与えられた単語列に対して，ID番号の列を返す関数を実装せよ．
# ただし，出現頻度が2回未満の単語のID番号はすべて0とせよ．

import numpy as np
import pandas as pd
import joblib
import re
import torch
from collections import defaultdict
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

df_news = pd.read_csv('/content/drive/My Drive/2020年度/勉強会/newsCorpora.csv', sep='\t', header=None)
df_news.columns = ['ID', 'Title', 'URL', 'Publisher', 'Category', 'Story', 'Hostname', 'Timestamp']
df_extracted = df_news[df_news['Publisher'].isin(["Huffington Post"])]
df_extracted.sample(frac=1, random_state=0)

y = df_extracted['Category']
X = df_extracted[['Title', 'Category']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=0)

y_train = y_train.map({'b': 0, 'e': 1, 't': 2, 'm': 3}).values
y_test = y_test.map({'b': 0, 'e': 1, 't': 2, 'm': 3}).values
y_valid = y_valid.map({'b': 0, 'e': 1, 't': 2, 'm': 3}).values

def preprocess(text):
  line = text.lower()
  cleaned = re.sub('[^0-9a-zA-Z]+',' ', line) # remove non-alphanumeric
  cleaned = ' '.join([i for i in cleaned.split() if not i in stop_words]) # remove stopwords
  return cleaned

phi = defaultdict(int)
for text in X_train['Title'].values:
  text = preprocess(text)
  for word in text.split():
    phi[word] += 1

phi = sorted(phi.items(), key=lambda x:x[1], reverse=True)
ids = {word: i+1 for i, (word, count) in enumerate(phi) if count >1}

def sent2ids(text, dic):
  id = []
  for word in text.split():
    if word in dic:
      id.append(dic[word])
    else:
      id.append(0)
  return id

for text in X_train['Title'].values:
  text = preprocess(text)
  text = sent2ids(text, ids)
  print(text)