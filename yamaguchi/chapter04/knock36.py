# 「defaultdict」をインポート
from collections import defaultdict
# 可視化ライブラリ「matplotlib」と日本語対応の「japanize_matplotlib」をインポート
import matplotlib.pyplot as plot
import japanize_matplotlib

file = 'neko.txt.mecab'

def mecab(nuko):
    result = []

    for line in nuko.split('\n'):
        if line == '':
            return result
        (surface, tmp) = line.split('\t') # 水平タブ
        tmp = tmp.split(',')
        # 出力フォーマットは
        # 「表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,
        # 活用形,原形,読み,発音」の順
        d = {
            'surface': surface,
            'base': tmp[6],
            'pos': tmp[0],
            'pos1': tmp[1]
        }
        result.append(d)

# 「単語，出現回数」のペアの辞書を作成
def dictionary(nuko):
    return [i['base'] + '_' + i['pos'] + '_' + i['pos1'] for i in nuko]

def neko():
    with open(file, encoding='utf-8') as f:
        # 「split()」でデータを分割
        text = f.read().split('EOS\n')
    # 不要な行を「filter()」で除去
    text = list(filter(lambda x: x != '', text))
    text = [mecab(nuko) for nuko in text]
    wordlist = [dictionary(nuko) for nuko in text]

    dic = defaultdict(int)

    for word in wordlist:
        for w in word:
            dic[w] += 1

    answer = sorted(dic.items(), key = lambda x: x[1], reverse=True)[:10]
    labels = [j[0] for j in answer]
    values = [j[1] for j in answer]

    # 棒グラフを表示
    plot.figure(figsize=(10, 10))
    plot.barh(labels, values)
    plot.show()

if __name__ == '__main__':
    neko()
