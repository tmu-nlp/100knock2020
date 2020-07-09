from gensim.models import KeyedVectors
import numpy as np
import joblib
import nltk
import re
import torch
from sklearn.model_selection import train_test_split

df_news = pd.read_csv('newsCorpora.csv', sep='\t', header=None)
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

nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

def preprocess(text):
  line = text.lower()
  cleaned = re.sub('[^0-9a-zA-Z]+',' ', line) # remove non-alphanumeric
  cleaned = ' '.join([i for i in cleaned.split() if not i in stop_words]) # remove stopwords
  return cleaned

def vectorize(text):
  sent = preprocess(text)
  vec = [model[word] for word in sent.split() if word in model.vocab]
  if len(vec) > 0:
    return torch.tensor(sum(vec)/len(vec))
  else:
    return torch.empty(300)

X_train_torch = torch.stack([vectorize(text) for text in X_train['Title'].values])
X_valid_torch = torch.stack([vectorize(text) for text in X_valid['Title'].values])