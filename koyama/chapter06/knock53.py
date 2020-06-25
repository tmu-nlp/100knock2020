# 53. 予測
# 52で学習したロジスティック回帰モデルを用い，与えられた記事見出しからカテゴリとその予測確率を計算するプログラムを実装せよ．

from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

if __name__ == "__main__":
    # データを読み込む
    X_test = pd.read_table("test.feature.txt", header=None)
    Y_test = pd.read_table("test.txt", header=None)[1]

    # モデルを読み込む
    clf = joblib.load("model.joblib")

    # 予測を行う
    Y_predict = clf.predict(X_test)     # 予測したカテゴリのリスト
    Y_proba = clf.predict_proba(X_test) # それぞれのカテゴリである確率のリスト

    # 予測したカテゴリとそれぞれのカテゴリである確率を表示する
    for i, proba in enumerate(Y_proba):
        print(f"predict:{Y_predict[i]}\tb:{proba[0]} e:{proba[1]} m:{proba[2]} t:{proba[3]}")