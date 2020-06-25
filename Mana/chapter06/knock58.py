import pandas as pd
from sklearn import __version__ as sklearn_version
#from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV


df_news = pd.read_csv('newsCorpora.csv', sep='\t', header=None)
df_news.columns = ['ID', 'Title', 'URL', 'Publisher', 'Category', 'Story', 'Hostname', 'Timestamp']

df_extracted = df_news[df_news['Publisher'].isin(["Huffington Post"])]
df_extracted.sample(frac=1, random_state=0)

y = df_extracted['Category']
X = df_extracted[['Title', 'Category']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=0)

#############################################

tfidf = TfidfVectorizer()

param_grid = [{'clf__C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           verbose=1,
                           n_jobs=-1)

gs_lr_tfidf.fit(X_train['Title'].values, y_train.values)
#print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
#print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
#print('Test Accuracy: %.3f' % gs_lr_tfidf.score(X_test['Title'].values, y_test.values))

y_predict = gs_lr_tfidf.predict(X_test['Title'].values)
#print(classification_report(y_test.values, y_predict))
print(gs_lr_tfidf.cv_results_)

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