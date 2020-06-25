# 別ファイルのプログラムをインポート
from chapter06 import knock50
from chapter06 import knock51
from chapter06 import knock52

# 記事見出しの指定
dic = {'b': 'business', 't': 'science and technology', 'e': 'entertainment', 'm': 'health'}

# knock52で学習したロジスティック回帰モデルを用いて，
# 与えられた記事見出しからカテゴリとその予測確率を計算する．
def predict(text):
    text = [text]
    X = knock51.value.transform(text)
    # 「predict_proba(X)」では，各データがそれぞれのクラスに所属する確率を返す．
    ls_proba = knock52.logistic.predict_proba(X)
    for proba in ls_proba:
        for c, p in zip(knock52.logistic.classes_, proba):
            print (dic[c]+':',p)

# カテゴリとその予測確率を計算したものを出力
s = knock50.train_df.iloc[0]['TITLE']
print(s)
predict(s)
