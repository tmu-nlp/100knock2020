import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV


df_news = pd.read_csv('newsCorpora.csv', sep='\t', header=None)
df_news.columns = ['ID', 'Title', 'URL', 'Publisher', 'Category', 'Story', 'Hostname', 'Timestamp']

le = LabelEncoder()
le = le.fit(df_news['Category'])
df_news['Category'] = le.transform(df_news['Category'])

df_extracted = df_news[df_news['Publisher'].isin(["Huffington Post"])]
df_extracted.sample(frac=1, random_state=0)

y = df_extracted['Category']
X = df_extracted[['Title', 'Category']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=0)

#############################################

tfidf = TfidfVectorizer()

param_grid = [{'clf__penalty': ['l1', 'l2'], 'clf__C': [1.0, 10.0, 100.0]} ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(solver='liblinear', random_state=0))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)

gs_lr_tfidf.fit(X_train['Title'].values, y_train.values)
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
print('Test Accuracy: %.3f' % gs_lr_tfidf.score(X_test['Title'].values, y_test.values))

"""
Best parameter set: {'clf__C': 100.0, 'clf__penalty': 'l2'} 
CV Accuracy: 0.789
Test Accuracy: 0.825
"""