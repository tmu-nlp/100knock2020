# 66. WordSimilarity-353での評価
# The WordSimilarity-353 Test Collectionの評価データをダウンロードし，単語ベクトルにより計算される類似度のランキングと，
# 人間の類似度判定のランキングの間のスピアマン相関係数を計算せよ．

from gensim.models import KeyedVectors
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    # データを読み込む
    model = model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
    df = pd.read_csv("combined.csv") # Word 1,Word 2,Human (mean)

    sim = []
    for i in tqdm(range(len(df))):
        line = df.iloc[i] # i行目を取り出す
        cos_sim = model.similarity(line["Word 1"], line["Word 2"])
        sim.append(cos_sim)
    df["w2v"] = sim

    corr = df[["Human (mean)", "w2v"]].corr(method="spearman")

    print(corr)


# 結果
#               Human (mean)       w2v
# Human (mean)      1.000000  0.700017
# w2v               0.700017  1.000000