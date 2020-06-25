# データ解析のライブラリをインポート
import pandas as pd
# scikit-learnでデータを訓練用とテスト用に分割できるようにする
from sklearn.model_selection import train_test_split
# 簡潔に記述できるよう，高階関数をインポート
from functools import reduce

# 情報源(publisher)が"Reuters"，"Huffington Post"，"Businessweek"，
# "Contactmusic.com"，"Daily Mail"の事例(記事)のみを抽出
# csvファイルの読み込み
news = pd.read_csv('NewsAggregatorDataset/newsCorpora.csv', sep='\t', header=None)
# カラム名を指定
news.columns = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
# 情報源(publisher)を指定
publisher = ['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']
# 指定した情報源の事例(記事)のみを抽出
article = [news.PUBLISHER == i for i in publisher]
ARTICLE =reduce(lambda a, b: a | b, article)
df = news[ARTICLE]

# 抽出した事例(記事)をランダムに並び替え
# 全てをサンプリングするので，並び替えと等価．
df = df.sample(frac=1)

# 抽出された事例(記事)の80%を学習データ，残りの10%ずつを検証データと評価データに分割する．
# 8:2としている
train_df, valid_test_df = train_test_split(df, test_size=0.2) # 8:2
valid_df, test_df = train_test_split(valid_test_df, test_size=0.5) # 8:1:1

# それぞれ「train.txt」，「valid.txt」，「test.txt」というファイル名で保存する．
train_df.to_csv('train.txt', columns = ['CATEGORY', 'TITLE'], sep='\t',header=False, index=False)
valid_df.to_csv('valid.txt', columns = ['CATEGORY', 'TITLE'], sep='\t',header=False, index=False)
test_df.to_csv('test.txt', columns = ['CATEGORY', 'TITLE'], sep='\t',header=False, index=False)

# 事例数の確認
df['CATEGORY'].value_counts()
