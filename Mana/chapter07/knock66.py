#The WordSimilarity-353 Test Collectionの評価データをダウンロードし，
# 単語ベクトルにより計算される類似度のランキングと，
# 人間の類似度判定のランキングの間のスピアマン相関係数を計算せよ．

from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin", binary=True
)

print(model.evaluate_word_pairs("combined.tab"))


#(0.6238773466616107, 1.7963237724171284e-39),
# SpearmanrResult
# (correlation=0.6589215888009288, 
# pvalue=2.5346056459149263e-45),
# 0.0)