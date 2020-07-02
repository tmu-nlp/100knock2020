# 学習済み単語ベクトルを扱うためにインポート
import gensim
# デンドログラムを作成するためにインポート
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

def main():
    # 「word2vec」で単語をベクトル化する
    # 「save_word2vec_format」で保存したモデルを「load_word2vec_format」で読み込む
    # 「KeyedVectors」で追加学習に必要なデータを除いてWord2Vecのモデルを軽量化
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    # 「United States」は内部的に「United_States」と表現されている
    model['United_States']

    plt.figure(figsize=(32.0, 24.0))
    link = linkage(vec, method='ward')
    dendrogram(link, labels=target_countries,leaf_rotation=90,leaf_font_size=10)
    plt.show()

if __name__ == '__main__':
    main()
