# 50. データの入手・整形
# News Aggregator Data Setをダウンロードし、以下の要領で学習データ（train.txt），検証データ（valid.txt），評価データ（test.txt）を作成せよ．
# ダウンロードしたzipファイルを解凍し，readme.txtの説明を読む．
# 情報源（publisher）が”Reuters”, “Huffington Post”, “Businessweek”, “Contactmusic.com”, “Daily Mail”の事例（記事）のみを抽出する．
# 抽出された事例をランダムに並び替える．
# 抽出された事例の80%を学習データ，残りの10%ずつを検証データと評価データに分割し，それぞれtrain.txt，valid.txt，test.txtというファイル名で保存する．
# ファイルには，１行に１事例を書き出すこととし，カテゴリ名と記事見出しのタブ区切り形式とせよ（このファイルは後に問題70で再利用する）．
# 学習データと評価データを作成したら，各カテゴリの事例数を確認せよ．

from sklearn.model_selection import train_test_split
import pandas as pd
import collections

if __name__ == "__main__":
    # データを読み込む
    newsCorpora_path = "newsCorpora.csv"
    newsCorpora = pd.read_csv(newsCorpora_path, header=None, sep="\t")

    # 列の名前を設定する
    colums_name = ["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"]
    newsCorpora.columns = colums_name

    # PUBLISHERが、”Reuters”、“Huffington Post”、“Businessweek”、“Contactmusic.com”、“Daily Mail” の事例のみを抽出する
    newsCorpora = newsCorpora[newsCorpora["PUBLISHER"].isin(["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"])]

    # 抽出された事例をランダムに並び替える
    # frac: 抽出する行・列の割合を指定
    # random_state: 乱数シードの固定
    newsCorpora = newsCorpora.sample(frac=1, random_state=0)

    # X = "TITLE" から Y = "CATEGORY" を予測する
    X = newsCorpora["TITLE"]
    Y = newsCorpora["CATEGORY"]

    # train:valid:test = 8:1:1 にしたい
    # まず、全体を train:(valid + test) = 8:2 に分ける
    # 次に、(valid + test) を valid:test = 5:5 に分ける
    # stratify: 層化抽出（元のデータの比率と同じになるように分ける）
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)
    X_valid, X_test, Y_valid, Y_test = train_test_split(X_test, Y_test, test_size=0.5, stratify=Y_test, random_state=0)

    # X_train と Y_train を列方向に連結する
    # axis: 連結方向
    XY_train = pd.concat([X_train, Y_train], axis=1)
    XY_valid = pd.concat([X_valid, Y_valid], axis=1)
    XY_test = pd.concat([X_test, Y_test], axis=1)

    # csvファイルとして保存する
    XY_train.to_csv("train.txt", sep="\t", index=False, header=None)
    XY_valid.to_csv("valid.txt", sep="\t", index=False, header=None)
    XY_test.to_csv("test.txt", sep="\t", index=False, header=None)

    # 学習データ、検証データ、評価データの事例数を確認する
    print(collections.Counter(Y_train)) # Counter({'b': 4502, 'e': 4223, 't': 1219, 'm': 728})
    print(collections.Counter(Y_valid)) # Counter({'b': 562, 'e': 528, 't': 153, 'm': 91})
    print(collections.Counter(Y_test))  # Counter({'b': 563, 'e': 528, 't': 152, 'm': 91})


