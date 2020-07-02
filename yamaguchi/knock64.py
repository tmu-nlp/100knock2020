# 学習済み単語ベクトルを扱うためにインポート
import gensim

def main():
    # 「word2vec」で単語をベクトル化する
    # 「save_word2vec_format」で保存したモデルを「load_word2vec_format」で読み込む
    # 「KeyedVectors」で追加学習に必要なデータを除いてWord2Vecのモデルを軽量化
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    # 「United States」は内部的に「United_States」と表現されている
    model['United_States']

    # ダウンロードした単語アナロジーの評価データを開く
    with open('evaluation_data.txt') as f:
        analogies = f.readlines()

    # 結果をファイルに保存
    with open('knock64.txt', mode='w', encoding="utf-8") as f:
        # 「enumerate()」でfor文の中でインデックスを取得
        for i, analogy in enumerate(analogies):
            words = analogy.split()
            # 「most_similar」を用いて類似度が最も高い単語とその類似度を出力
            # 「positive」と「negative」を指定して，
            # 2列目の単語に3列目の単語を足し，そこから1列目の単語を引く．
            # 求めた単語と類似度は各事例の末尾に追記
            if len(words) == 4:
                ans = model.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=1)[0]
                words += [ans[0], str(ans[1])]
                output = ' '.join(words) + '\n'
            else:
                output = analogy

            f.write(output)
            if (i % 100 == 0):
                print(i)

if __name__ == '__main__':
    main()
