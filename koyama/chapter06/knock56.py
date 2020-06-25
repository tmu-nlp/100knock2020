# 56. 適合率，再現率，F1スコアの計測
# 52で学習したロジスティック回帰モデルの適合率，再現率，F1スコアを，評価データ上で計測せよ．
# カテゴリごとに適合率，再現率，F1スコアを求め，カテゴリごとの性能をマイクロ平均（micro-average）とマクロ平均（macro-average）で統合せよ．

from sklearn.metrics import recall_score, precision_score, f1_score
import pandas as pd
import joblib

if __name__ == "__main__":
    # データを読み込む
    X_test = pd.read_table("test.feature.txt", header=None)
    Y_test = pd.read_table("test.txt", header=None)[1]

    # モデルを読み込む
    clf = joblib.load("model.joblib")

    # カテゴリを予測する
    Y_pred = clf.predict(X_test)

    # 適合率、再現率、F1スコアを求める
    rec = recall_score(Y_test, Y_pred, average=None)
    prec = precision_score(Y_test, Y_pred, average=None)
    f1 = f1_score(Y_test, Y_pred, average=None)

    # 適合率、再現率、F1スコアのカテゴリごとのマイクロ平均、マクロ平均を求める
    rec_micro = recall_score(Y_test, Y_pred, average="micro")
    rec_macro = recall_score(Y_test, Y_pred, average="macro")
    prec_micro = precision_score(Y_test, Y_pred, average="micro")
    prec_macro = precision_score(Y_test, Y_pred, average="macro")
    f1_micro = f1_score(Y_test, Y_pred, average="micro")
    f1_macro = f1_score(Y_test, Y_pred, average="macro")

    # 結果を表示する
    print(f"適合率  :{rec}\tマイクロ平均:{rec_micro}\tマクロ平均:{rec_macro}")
    print(f"再現率  :{prec}\tマイクロ平均:{prec_micro}\tマクロ平均:{prec_macro}")
    print(f"F1スコア:{f1}\tマイクロ平均:{f1_micro}\tマクロ平均:{f1_macro}")

# 結果
# 適合率  :[0.93783304 0.97727273 0.6043956  0.61842105]  マイクロ平均:0.8943028485757122 マクロ平均:0.7844806054000222
# 再現率  :[0.89795918 0.88965517 0.94827586 0.87037037]  マイクロ平均:0.8943028485757122 マクロ平均:0.9015651471316495
# F1スコア:[0.91746308 0.93140794 0.73825503 0.72307692]  マイクロ平均:0.8943028485757122 マクロ平均:0.827550743614671