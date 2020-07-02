# 学習済み単語ベクトルを扱うためにインポート
import gensim

def main():
    # 「word2vec」で単語をベクトル化する
    # 「save_word2vec_format」で保存したモデルを「load_word2vec_format」で読み込む
    # 「KeyedVectors」で追加学習に必要なデータを除いてWord2Vecのモデルを軽量化
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    # 「United States」は内部的に「United_States」と表現されている
    model['United_States']

    # 結果をファイルに保存
    # 「most_similar」を用いて「United States」と類似度が高い10語とその類似度を出力
    with open('knock62.txt', mode='w', encoding="utf-8") as f:
        print(model.most_similar('United_States', topn=10), file=f)

if __name__ == '__main__':
    main()
