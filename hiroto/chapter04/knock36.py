'''
36. 頻度上位10語Permalink
出現頻度が高い10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ．
'''
import collections
import matplotlib.pyplot as plt
import japanize_matplotlib

#文章全体の単語のリストを返す
def words_list():
    with open('neko.txt.mecab', encoding="utf-8_sig") as f:
        words = []
        for line in f:
            line_processed = line.strip('\n')
            if line_processed == 'EOS':
                break
            else:
                columns = line_processed.split('\t')
                cols = columns[1].split(',')
                #cols[6]:基本形
                words.append(cols[6])
    return words

def frequency(words):
    #Counterオブジェクト
    freq = collections.Counter(words)
    #most_common(n):要素が大きいものからn個のタプルのリストを返す。
    return freq.most_common()

def main():
    freq = frequency(words_list())
    #x:単語, y:出現頻度
    x, y = [], []
    cnt = 0
    #上位10単語だけ
    while(cnt < 10):
        x.append(freq[cnt][0])
        y.append(freq[cnt][1])
        cnt += 1

    plt.bar(range(len(x)), y, align = 'center')
    plt.xticks(range(len(x)), x)
    plt.xlabel('単語')
    plt.ylabel('出現頻度')
    plt.show()

if __name__ == "__main__":
    main()
