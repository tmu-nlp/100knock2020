# 別ファイルのプログラムをインポート
from chapter06 import knock50
from chapter06 import knock51

# 「scikit-learn」に実装されているものを利用
from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression()
logistic.fit(knock51.train_value, knock50.train_df['CATEGORY'])
