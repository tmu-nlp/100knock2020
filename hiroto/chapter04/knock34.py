'''
34. 名詞の連接Permalink
名詞の連接（連続して出現する名詞）を最長一致で抽出せよ．
'''
def mapping():
    with open('neko.txt.mecab') as f:
        morphemes_list = []
        morphemes = []
        for line in f:
            line_processed = line.strip('\n')
            if line_processed == 'EOS':
                break
            else:
                columns = line_processed.split('\t')
                cols = columns[1].split(',')
                morpheme = {}
                morpheme['surface'] = columns[0]
                morpheme['base'] = cols[6]
                morpheme['pos'] = cols[0]
                morpheme['pos1'] = cols[1]
                morphemes.append(morpheme)
                if morpheme['pos1'] == '句点':
                    morphemes_list.append(morphemes)
                    morphemes = []
                else : pass
    return morphemes_list

def extract_connection(morphemes_list):
    cnt = 0
    max_cnt = 0
    #名詞の連接を一時保存
    temp_connection = ''
    #各，名詞の連接とそれに含まれる単語数の辞書
    connections = {}
    for morphemes in morphemes_list:
        for i in range(len(morphemes)):
            if morphemes[i]['pos'] == '名詞':
                cnt += 1
                temp_connection += morphemes[i]['surface']
            else:
                if temp_connection != '':
                    connections[temp_connection] = cnt
                    temp_connection = ''
                    cnt = 0
                else: pass
                #最長のやつをとる
                '''
                if cnt > max_cnt:
                    max_cnt = cnt
                    longest_connections[temp_connection] = max_cnt
                else: temp_connection = ''
                cnt = 0
                '''
    return connections

def main():
    connections = extract_connection(mapping())
    #連接している単語数が多い順にソート
    connections_sorted = sorted(connections.items()\
        , reverse = True, key = lambda x : x[1])
    #最初の10個表示
    print(connections_sorted[0:10])


if __name__ == "__main__":
    main()

'''
neko_parsed.txtの
一       1
人間中   6
四五遍   56
我等猫族         108
壱円五十銭       252
一杯一杯一杯     442
三毛子さん三毛子さん     649
manyaslip'twixtthecupandthelip   1139
'''
