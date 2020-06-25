#適合率、再現率、F1スコア
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

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

Y_pred = lr.predict(X_test_vec)

print(classification_report(y_test.values, Y_pred))

"""
              precision    recall  f1-score   support

           b       0.81      0.68      0.74        44
           e       0.82      0.95      0.88       123
           m       0.81      0.69      0.75        32
           t       0.80      0.68      0.74        47

    accuracy                           0.82       246
   macro avg       0.81      0.75      0.78       246
weighted avg       0.82      0.82      0.81       246

"""