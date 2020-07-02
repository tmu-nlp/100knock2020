# 別ファイルのプログラムをインポート
from chapter06 import knock50
from chapter06 import knock51
from chapter06 import knock52

# クラス分類結果を評価する(正解率を求める)ためにインポート
from sklearn.metrics import accuracy_score

# ロジスティック回帰モデルの正解率を学習データ及び評価データ上で計測
train_pred_correct = knock52.logistic.predict(knock51.train_value)
test_pred_correct = knock52.logistic.predict(knock51.test_value)
train_correct = knock50.train_df['CATEGORY']
test_correct = knock50.test_df['CATEGORY']

# 評価結果を出力
print (accuracy_score(train_correct, train_pred_correct))
print (accuracy_score(test_correct, test_pred_correct))
