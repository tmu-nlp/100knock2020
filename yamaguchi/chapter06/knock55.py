# 別ファイルのプログラムをインポート
from chapter06 import knock54

# 混同行列の作成のためにインポート
from sklearn.metrics import confusion_matrix

# 「confusion_matrix」では，明示的にラベルの順序を指定できる．
# knock52で学習したロジスティック回帰モデルの混同行列を学習データ及び評価データ上で作成する
# 'b': 'business', 't': 'science and technology', 'e': 'entertainment', 'm': 'health'
print(confusion_matrix(knock54.train_correct, knock54.train_pred_correct, labels = ['b', 't', 'e', 'm']))
print(confusion_matrix(knock54.test_correct, knock54.test_pred_correct, labels = ['b', 't', 'e', 'm']))
