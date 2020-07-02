# 別ファイルのプログラムをインポート
from chapter06 import knock54

# 「precision_score」,「recall_score」,「f1_score」を用いて，
# knock52で学習したロジスティック回帰モデルの適合率，再現率，F1スコアを評価データ上でカテゴリごとに計測する．
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# 引数「average」で「None」を指定した場合はリストで返却，
print (precision_score(knock54.test_correct, knock54.test_pred_correct, average=None, labels=['b', 't', 'e', 'm']))
print (recall_score(knock54.test_correct, knock54.test_pred_correct, average=None, labels=['b', 't', 'e', 'm']))
print (f1_score(knock54.test_correct, knock54.test_pred_correct, average=None, labels=['b', 't', 'e', 'm']))

# 引数「average」で「micro」を指定した場合はマイクロ平均が返却される．
print (precision_score(knock54.test_correct, knock54.test_pred_correct, average='micro', labels=['b', 't', 'e', 'm']))
print (recall_score(knock54.test_correct, knock54.test_pred_correct, average='micro', labels=['b', 't', 'e', 'm']))
print (f1_score(knock54.test_correct, knock54.test_pred_correct, average='micro', labels=['b', 't', 'e', 'm']))

# 引数「average」で「macro」を指定した場合はマクロ平均が返却される．
print (precision_score(knock54.test_correct, knock54.test_pred_correct, average='macro', labels=['b', 't', 'e', 'm']))
print (recall_score(knock54.test_correct, knock54.test_pred_correct, average='macro', labels=['b', 't', 'e', 'm']))
print (f1_score(knock54.test_correct, knock54.test_pred_correct, average='macro', labels=['b', 't', 'e', 'm']))
