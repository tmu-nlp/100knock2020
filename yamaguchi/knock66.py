# 学習済み単語ベクトルを扱うためにインポート
import gensim
# スピアマンの順位相関係数を計算するためにインポート
import pandas as pd

def main():
    # 「word2vec」で単語をベクトル化する
    # 「save_word2vec_format」で保存したモデルを「load_word2vec_format」で読み込む
    # 「KeyedVectors」で追加学習に必要なデータを除いてWord2Vecのモデルを軽量化
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    # 「United States」は内部的に「United_States」と表現されている
    model['United_States']

    # 「The WordSimilarity-353 Test Collection」の評価データを読み込む
    deta = pd.read_csv('wordsim353/combined.csv')
    sim = []

    # 単語ベクトルにより計算される類似度のランキングと，
    # 人間の類似度判定のランキングとの間のスピアマン相関係数を計算．
    for i in range(len(deta)):
        line = deta.iloc[i]
        sim.append(model.similarity(line['Word 1'], line['Word 2']))
    deta['w2v'] = sim
    deta[['Human (mean)', 'w2v']].corr(method='spearman')

    # 結果をファイルに保存
    with open('knock66.txt', mode='w', encoding="utf-8") as f:
        print(deta[['Human (mean)', 'w2v']].corr(method='spearman'), file=f)

if __name__ == '__main__':
    main()
