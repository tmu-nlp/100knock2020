# 57. 特徴量の重みの確認
# 52で学習したロジスティック回帰モデルの中で，重みの高い特徴量トップ10と，重みの低い特徴量トップ10を確認せよ．

import joblib

if __name__ == "__main__":
    # データを読み込む
    clf = joblib.load("model.joblib")
    vocabulary_ = joblib.load("vocabulary_.joblib")

    # 特徴量の重みを得る
    # 各カテゴリごとに特徴量の重みが入っているというリストのリストになっている
    coefs = clf.coef_

    # カテゴリ名
    category_names = ["business", "entertainment", "health", "science and technology"]

    # 各カテゴリごとに重みの高い特徴量トップ10と、重みの低い特徴量トップ10を得る
    for i, category_features in enumerate(coefs):
        # keyを語彙、valueをその語彙の重みとする辞書を作る
        features = dict()
        for word, index in vocabulary_.items():
            features[word] = category_features[index]

        # 重みの高い特徴量トップ10を表示する
        print(f"Top 10 of \"{category_names[i]}\":")
        for word, weight in sorted(features.items(), key=lambda x:x[1], reverse=True)[:10]:
            print(f"{word:<10s} -> {weight}")
        print()

        # 重みの低い特徴量トップ10を表示する
        print(f"Worst 10 of \"{category_names[i]}\":")
        for word, weight in sorted(features.items(), key=lambda x:x[1], reverse=False)[:10]:
            print(f"{word:<10s} -> {weight}")
        print()

# 結果
# Top 10 of "business":
# bank       -> 3.527738556221488
# china      -> 3.3661180633006835
# fed        -> 3.3523500669410486
# ecb        -> 3.2255932044474758
# stocks     -> 3.1236255984471164
# euro       -> 2.9889222237472644
# update     -> 2.8488141752308436
# oil        -> 2.6689218405012136
# yellen     -> 2.6424869497500794
# profit     -> 2.5534699258845492
#
# Worst 10 of "business":
# and        -> -2.4098190288122114
# the        -> -1.9897545163813612
# she        -> -1.9204103288949663
# ebola      -> -1.9055334550191803
# her        -> -1.8664023667929035
# star       -> -1.778802837337786
# apple      -> -1.7639372743732091
# google     -> -1.761932005575555
# kardashian -> -1.6494700173611412
# facebook   -> -1.5978075399802616
#
# Top 10 of "entertainment":
# kardashian -> 3.355884232743939
# chris      -> 2.822619035936853
# star       -> 2.570725306755418
# kim        -> 2.5379052703968834
# cyrus      -> 2.472045538692194
# miley      -> 2.468956611943708
# she        -> 2.4622014878352467
# her        -> 2.457630827390672
# film       -> 2.297429051124015
# paul       -> 2.1820021917130394
#
# Worst 10 of "entertainment":
# update     -> -3.688748714627132
# us         -> -3.254474218517357
# google     -> -2.786667340591098
# facebook   -> -2.34204419486849
# china      -> -2.3197982300740914
# gm         -> -2.2188860007612683
# ceo        -> -2.1151861875476445
# billion    -> -2.025479136618963
# says       -> -1.9866619635465652
# apple      -> -1.9593494658588502
#
# Top 10 of "health":
# ebola      -> 4.567486685590472
# drug       -> 3.866085460690856
# fda        -> 3.7056252753220225
# cancer     -> 3.6323808099281703
# study      -> 3.1293749759819813
# mers       -> 2.867835313275537
# could      -> 2.44738338534489
# virus      -> 2.4393002556042784
# health     -> 2.409984005316182
# heart      -> 2.3077102597385792
#
# Worst 10 of "health":
# gm         -> -1.123341886220394
# ceo        -> -1.015147704980903
# facebook   -> -0.9728939809530622
# apple      -> -0.9623416114867839
# google     -> -0.9043242327296093
# amazon     -> -0.8918391631125068
# climate    -> -0.8885971881983962
# at         -> -0.8478354648468098
# deal       -> -0.8128483969242937
# bank       -> -0.8045776786503791
#
# Top 10 of "science and technology":
# google     -> 5.452923578896238
# facebook   -> 4.912745715801806
# apple      -> 4.68562835171882
# climate    -> 4.01428334861419
# microsoft  -> 3.856984887802714
# gm         -> 3.1573622604100677
# tesla      -> 3.108222645714443
# nasa       -> 2.785146438984926
# comcast    -> 2.717500512058476
# fcc        -> 2.4453115731758412
#
# Worst 10 of "science and technology":
# stocks     -> -1.4220184554548991
# drug       -> -1.1442679237215607
# fed        -> -1.0683139026715838
# her        -> -1.048415583718837
# ecb        -> -0.9937642215348839
# american   -> -0.9860239884859654
# kardashian -> -0.9632007754068459
# day        -> -0.9415411507102482
# not        -> -0.9365104413821095
# but        -> -0.932724979466381
