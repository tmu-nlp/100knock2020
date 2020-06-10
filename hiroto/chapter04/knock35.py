'''
35. 単語の出現頻度Permalink
文章中に出現する単語とその出現頻度を求め，出現頻度の高い順に並べよ．
'''
import collections

#文章中に出てくる動詞の基本形を格納したリストを返す
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
    cnt = collections.Counter(words)
    #most_common(n):要素が大きいものからn個のタプルのリストを返す。
    return cnt.most_common()

def main():
     print(frequency(words_list()))

if __name__ == "__main__":
    main()
