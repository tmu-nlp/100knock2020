# 61. 単語の類似度
# “United States”と”U.S.”のコサイン類似度を計算せよ．

import numpy as np
from gensim.models import KeyedVectors

if __name__ == "__main__":
    # データを読み込む
    model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

    # コサイン類似度を計算する
    cos_sim = model.similarity("United_States", "U.S.")

    # 結果を表示する
    print(cos_sim)


# 結果
# 0.73107743
