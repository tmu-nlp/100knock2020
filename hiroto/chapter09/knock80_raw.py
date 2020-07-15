'''
80. ID番号への変換Permalink
問題51で構築した学習データ中の単語にユニークなID番号を付与したい．
学習データ中で最も頻出する単語に1，2番目に頻出する単語に2，……といった方法で，
学習データ中で2回以上出現する単語にID番号を付与せよ．そして，
与えられた単語列に対して，ID番号の列を返す関数を実装せよ．
ただし，出現頻度が2回未満の単語のID番号はすべて0とせよ．
'''
import spacy
import pickle
from tqdm import tqdm
import pandas as pd
from collections import Counter
import torch
data_dir = "./data_raw"
dic_dir = "./dic_raw"
col_names = ['CATEGORY', 'TITLE']
nlp = spacy.load('en_core_web_sm')
train_fname = f"{data_dir}/train.txt"
valid_fname = f"{data_dir}/valid.txt"
test_fname = f"{data_dir}/test.txt"

def tokenize(sentence):
    sentence = nlp(sentence)
    tokens = [token.lemma_.lower() for token in sentence]
    return tokens

def counter(tokens, cnter):
    cnt = Counter(tokens)
    cnter += cnt
    return cnter

def preprocess(text):
    cnter = Counter()
    for sentence in text:
        tokens = tokenize(sentence)
        cnter = counter(tokens, cnter)
    #単語からIDへの変換辞書
    dic = {}
    #ID化するための辞書を作成
    cnt = 1
    for item in cnter.most_common():
        #頻度が２回未満の単語のIDを0にする　
        if item[1] < 2:
            dic[item[0]] = 0
        else: dic[item[0]] = cnt
        cnt += 1
    return dic #頻度が降順になっている
    
def word_to_id(sentence, dic):
    tokens = tokenize(sentence)
    sentence_id = [dic[token] for token in tokens]
    return sentence_id


def main():
    df = pd.read_csv(train_fname, header=None, names=col_names, sep='\t')
    dic = preprocess(df['TITLE'].values)
    
    for sentence in tqdm(df['TITLE'].values):
        print(word_to_id(sentence, dic))
    
    torch.save(dic, f"{dic_dir}/id_map.dic")

if __name__ == '__main__':
    main()

'''
[282, 1, 26, 1713, 3817, 2385, 2668, 2421]
[321, 868, 55, 1108, 3851, 1957, 6312, 164]
[46, 49, 2, 0, 4557, 102, 32, 6558]
[1045, 27, 3866, 25, 387, 24, 797, 17, 3179, 650]
[6235, 6236, 0, 2, 145, 1033, 2, 4002, 0, 301]
[1060, 1927, 2, 964, 3451, 349, 526, 4378, 6506, 605, 697, 5, 4]
[247, 670, 625, 1203, 351, 131, 0]
[338, 169, 4887, 0, 4924, 217, 2434, 0]
[3726, 3727, 1607, 2130, 566, 4785, 1974]
[2, 3064, 2, 179, 687, 727, 249, 369]
[234, 1902, 127, 2783, 50, 611, 206]
[1373, 1193, 100, 212, 568, 620, 3021]
[144, 3872, 2597, 367, 1701, 1890]
'''