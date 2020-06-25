import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df_news = pd.read_csv('newsCorpora.csv', sep='\t', header=None)
df_news.columns = ['ID', 'Title', 'URL', 'Publisher', 'Category', 'Story', 'Hostname', 'Timestamp']

df_extracted = df_news[df_news['Publisher'].isin(["Huffington Post"])]
df_extracted.sample(frac=1, random_state=0)

y = df_extracted['Category']
X = df_extracted[['Title', 'Category']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=0)

tfidf = TfidfVectorizer(use_idf=True, smooth_idf=True)
X_train_vec = tfidf.fit_transform(X_train['Title'].values)
X_test_vec = tfidf.transform(X_test['Title'].values)

#インスタンス生成
lr = LogisticRegression(C=100.0, random_state=0)
#適合
lr.fit(X_train_vec, y_train.values)

for c, coef in zip(lr.classes_, lr.coef_):
    print(c, coef)