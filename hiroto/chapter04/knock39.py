'''
39. Zipfの法則Permalink
単語の出現頻度順位を横軸，その出現頻度を縦軸として，両対数グラフをプロットせよ．
'''
import collections
import matplotlib.pyplot as plt
import japanize_matplotlib

#文章全体の単語のリストを返す
def words_list():
    with open('neko.txt.mecab', encoding="utf-8_sig") as file:
        words = []
        for line in file:
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
    freqs = collections.Counter(words)
    #most_common(n):要素が大きいものからn個のタプルのリストを返す。
    return freqs.most_common()

def main():
    #freqsは降順（単語, 出現頻度）
    freqs = frequency(words_list())
    #x:単語, y:出現頻度
    x, y = [], []
    cnt = 0
    while(cnt < len(freqs)):
        x.append(cnt + 1)
        y.append(freqs[cnt][1])
        cnt += 1

    plt.plot(x, y)
    plt.title("Zipf's law", fontsize = 20)
    plt.xticks(x)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('出現頻度順位')
    plt.ylabel('出現頻度')
    plt.grid(which = "both")
    plt.show()

if __name__ == "__main__":
    main()
