'''
66. WordSimilarity-353での評価Permalink
The WordSimilarity-353 Test Collectionの評価データをダウンロードし，
単語ベクトルにより計算される類似度のランキングと，
人間の類似度判定のランキングの間のスピアマン相関係数を計算せよ．
'''
import pickle
from scipy.stats import spearmanr

with open("./models/google_model.pickle", mode="rb") as f:
    model = pickle.load(f)

human, machine = [], []
with open("./wordsim353/combined.tab") as f:
    # 最初の行を飛ばす
    next(f)
    for line in f:
        cols = line.split()
        human.append(cols[2])
        score = model.similarity(cols[0], cols[1])
        machine.append(score)

spearman_corr = spearmanr(human, machine)[0]
print(f"spearman correlation: {spearman_corr}")

'''
SpearmanrResult(correlation=0.6849564489532377, pvalue=3.3287848950139166e-50)
spearman correlation: 0.6849564489532377
'''