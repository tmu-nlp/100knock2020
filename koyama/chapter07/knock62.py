# 62. 類似度の高い単語10件
# “United States”とコサイン類似度が高い10語と，その類似度を出力せよ．

from gensim.models import KeyedVectors

if __name__ == "__main__":
    # データを読み込む
    model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

    # "United_States"とコサイン類似度が高い10語を求める
    cos_top10 = model.most_similar("United_States", topn=10)

    # 結果を表示する
    for word, sim in cos_top10:
        print(f"{word}\t{sim}")


# 結果
# Unites_States	0.7877248525619507
# Untied_States	0.7541370391845703
# United_Sates	0.74007248878479
# U.S.	0.7310774326324463
# theUnited_States	0.6404393911361694
# America	0.6178410053253174
# UnitedStates	0.6167312264442444
# Europe	0.6132988929748535
# countries	0.6044804453849792
# Canada	0.6019070148468018
