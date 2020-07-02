# 別ファイルのプログラムをインポート
from chapter06 import knock50

# 特徴量抽出をする(単語の出現頻度を求める)ためにインポート
from sklearn.feature_extraction.text import CountVectorizer
# 数値計算のためのモジュールをインポート
import numpy as np

# 特徴量抽出をする(単語の出現頻度を求める)
value = CountVectorizer()
# 「fit_transform()」では「fit()」を実行した後に同じデータに対して「transform()」を実行する．
# 「fit()」では渡されたデータの最大値，最小値，平均，標準偏差，傾きなどの統計を取得し，内部メモリに保存する．
train_value = value.fit_transform(knock50.train_df['TITLE'])
# 「tranceform()」では「fit()」で取得した統計情報を使って，渡されたデータを実際に書き換える．
valid_value = value.transform(knock50.valid_df['TITLE'])
test_value = value.transform(knock50.test_df['TITLE'])
# 学習データの場合は，それ自体の統計を基に正規化や欠損地処理を行っても問題ないので，「fit_transform()」を用いて問題ない．
# 評価データの場合は，比較的データ数が少なく，学習データの統計を用いて正規化や欠損処理を行うべきなので，
# 学習データに対する「fit()」の結果で「transform()」を行う必要がある．

# それぞれ「train.feature.txt」，「valid.feature.txt」，「test.feature.txt」というファイル名で保存する．
# スパース行列(疎行列)から非スパース行列(密行列)に変換
np.savetxt('train.feature.txt', train_value.toarray(), fmt='%d')
np.savetxt('valid.feature.txt', valid_value.toarray(), fmt='%d')
np.savetxt('test.feature.txt', test_value.toarray(), fmt='%d')
