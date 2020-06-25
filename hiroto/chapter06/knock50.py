'''
ダウンロードしたzipファイルを解凍し，readme.txtの説明を読む．
情報源（publisher）が”Reuters”, “Huffington Post”, “Businessweek”, “Contactmusic.com”, 
“Daily Mail”の事例（記事）のみを抽出する．
抽出された事例をランダムに並び替える．
抽出された事例の80%を学習データ，残りの10%ずつを検証データと評価データに分割し，それぞれtrain.txt，valid.txt，
test.txtというファイル名で保存する．ファイルには，１行に１事例を書き出すこととし，カテゴリ名と記事見出しの
タブ区切り形式とせよ（このファイルは後に問題70で再利用する）．
'''
'''
FILENAME #1: newsCorpora.csv (102.297.000 bytes)
DESCRIPTION: News pages
FORMAT: ID \t TITLE \t URL \t PUBLISHER \t CATEGORY \t STORY \t HOSTNAME \t TIMESTAMP

where:
ID		Numeric ID
TITLE		News title 
URL		Url
PUBLISHER	Publisher name
CATEGORY	News category (b = business, t = science and technology, e = entertainment, m = health)
STORY		Alphanumeric ID of the cluster that includes news about the same story
HOSTNAME	Url hostname
TIMESTAMP 	Approximate time the news was published, as the number of milliseconds since the epoch 00:00:00 GMT, January 1, 1970
'''
import os
import pandas as pd
cwd = os.getcwd()
fname = f"{cwd}/NewsAggregatorDataset/newsCorpora.csv"
features = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
publishers = ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]

df = pd.read_csv(fname, sep='\t+', header=None, names = features, engine='python')
#df.query('PUBLISHER in ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]')
# X.isin(Y)でXのカラムがYの中に含まれているかをbool値で返してくれる
df = df[df['PUBLISHER'].isin(publishers)]
# fracでシャッフルして抽出する行の割合を設定するfeac=1=>100%
# seedを固定
df = df.sample(frac=1, random_state=0)
# drop=Trueだと元のインデックスの列'index'が削除される
df = df.reset_index(drop=True)
train_test_border = int(df.shape[0]*0.8)
train_df = df[0:train_test_border]
test_df = df[train_test_border:]
valid_test_border = int(test_df.shape[0]*0.5) + train_test_border
valid_df = df[train_test_border:valid_test_border]
test_df = df[valid_test_border:]

'''
一行に
CATEGORY\tTITLE
でそれぞれtrain.txt, valid.txt, test.txtに書き込む
'''
train_fname = f"{cwd}/data/train.txt"
valid_fname = f"{cwd}/data/valid.txt"
test_fname = f"{cwd}/data/test.txt"

def write_file(df, file):
    categories = df['CATEGORY'].values
    titles = df['TITLE'].values
    for category, title in zip(categories, titles):
        file.write(f'{category}\t{title}\n')

with open(train_fname, mode='w') as train_file\
    , open(valid_fname, mode='w') as valid_file\
    , open(test_fname, mode = 'w') as test_file:
    write_file(train_df, train_file)
    write_file(valid_df, valid_file)
    write_file(test_df, test_file)


print(train_df.shape)
print(valid_df.shape)
print(test_df.shape)
#wc -l train.txt
#wc -l valid.txt
#wc -l test.txt