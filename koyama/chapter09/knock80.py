# 80. ID番号への変換
# 問題51で構築した学習データ中の単語にユニークなID番号を付与したい．
# 学習データ中で最も頻出する単語に1，2番目に頻出する単語に2，……といった方法で，
# 学習データ中で2回以上出現する単語にID番号を付与せよ．
# そして，与えられた単語列に対して，ID番号の列を返す関数を実装せよ．
# ただし，出現頻度が2回未満の単語のID番号はすべて0とせよ．

from sklearn.model_selection import train_test_split
from collections import defaultdict
import pandas as pd
import string

def tokenizer(text, word2id, unk=0):
  ids = []
  for word in text.split():
      ids.append(word2id.get(word, unk))
  return ids

if __name__ == "__main__":
    # # データを読み込む
    train = pd.read_csv('train.txt', header=None, sep='\t', names=['TITLE', 'CATEGORY'])
    valid = pd.read_csv('valid.txt', header=None, sep='\t', names=['TITLE', 'CATEGORY'])
    test = pd.read_csv('test.txt', header=None, sep='\t', names=['TITLE', 'CATEGORY'])

    # 単語の頻度を集計する
    d = defaultdict(int)
    for text in train['TITLE']:
        for word in text.split():
            d[word] += 1
    d = sorted(d.items(), key=lambda x:x[1], reverse=True)

    # 単語ID辞書を作成する
    word2id = {}
    for i, (word, cnt) in enumerate(d):
        # 出現頻度が2回以上の単語を登録する
        if cnt <= 1:
            continue
        word2id[word] = i + 1

    # 結果を表示する
    text = train.iloc[1, train.columns.get_loc('TITLE')]
    print(f'テキスト: {text}')
    print(f'ID列: {tokenizer(text=text, word2id=word2id)}')

# 結果
# テキスト: GLOBAL MARKETS-Subdued Ifo takes M&A shine off European stocks
# ID列: [83, 0, 3253, 300, 854, 6653, 69, 192, 204]