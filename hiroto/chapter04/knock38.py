'''
38. ヒストグラムPermalink
単語の出現頻度のヒストグラムを描け．ただし，横軸は出現頻度を表し，1から単語の出現頻度の
最大値までの線形目盛とする．縦軸はx軸で示される出現頻度となった単語の異なり数（種類数）である．
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
    freqs = frequency(words_list())
    #count:出現頻度
    count = []
    cnt = 0
    while(cnt < len(freqs)):
        count.append(freqs[cnt][1])
        cnt += 1

    plt.hist(count, bins = 30, range = (1, 30))
    plt.xlim(xmin=1, xmax=30)
    plt.xlabel('出現頻度')
    plt.ylabel('単語の種類数')
    plt.show()

if __name__ == "__main__":
    main()
