# 別ファイルのプログラムをインポート
from chapter06 import knock51
from chapter06 import knock52
from chapter06 import knock53

names = knock51.np.array(knock51.value.get_feature_names())

# 'b': 'business', 't': 'science and technology', 'e': 'entertainment', 'm': 'health'
labels=['b', 't', 'e', 'm']

# knock52で学習したロジスティック回帰モデルの中で，
# 重みの高い特徴量と，重みの低い特徴量それぞれのトップ10を確認する．
# 「(インスタンス名).coef_」とすることで，パラメータ(重み)を取得できる．
# カテゴリ毎に表示する
for c, coef in zip(knock52.logistic.classes_, knock52.logistic.coef_):
    idx = knock51.np.argsort(coef)[::-1]
    print (knock53.dic[c])
    # 重みの高い特徴量トップ10
    print (names[idx][:10])
    # 重みの低い特徴量トップ10
    print (names[idx][-10:][::-1])
