#文単位の共起頻度
from collections import Counter
import matplotlib.pyplot as plt
import japanize_matplotlib

#sentence:1文中の単語の原形を集めた集合
#sentences:単語の原形で構成された文のリスト
def words_set():
    with open('neko.txt.mecab', encoding="utf-8_sig") as file:
        sentence = set()
        sentences = []
        for line in file:
            line_processed = line.strip('\n')
            if line_processed == 'EOS':
                break
            else:
                columns = line_processed.split('\t')
                cols = columns[1].split(',')
                #cols[6]:基本形
                sentence.add(cols[6])
                #cols[1]:品詞細分類
                if cols[1] == '句点':
                    sentences.append(sentence)
                    sentence = set()
                else : pass
    return sentences

'''
#（猫, X）のタプルのリストを作る
def mk_neko_pairs(words):
    neko_pairs = [('猫', word) for word in words]
    #neko_pairs.remove(('猫', '猫'))
    return neko_pairs
'''

def co_occurance_frequency(sentences):
    #cnt:単語のペアの出現頻度をカウントする
    cnt = Counter()
    for sentence in sentences:
        if ('猫' in sentence):
            temp = [('猫', word) for word in sentence]
            cnt += Counter(temp)
        else: pass

    #Counter型からlistに変換
    freqs = list(cnt.most_common())
    #freq[0]:単語のペア, freq[1]:共起頻度
    for freq in freqs:
        if freq[0] == ('猫', '猫'):
            freqs.remove(freq)

    return freqs

def main():
    sentences = words_set()
    freqs = co_occurance_frequency(sentences)

    x, y = [], []
    cnt = 0
    while(cnt < 10):
        x.append(freqs[cnt][0])
        y.append(freqs[cnt][1])
        cnt += 1

    plt.bar(range(len(x)), y, align = 'center')
    plt.xticks(range(len(x)), x, rotation = 30)
    plt.xlabel('単語のペア')
    plt.ylabel('共起頻度')
    plt.show()

    #print(freqs)

if __name__ == "__main__":
    main()
