'''
30. 形態素解析結果の読み込みPermalink
形態素解析結果（neko.txt.mecab）を読み込むプログラムを実装せよ．ただし，各形態素は
表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をキーとするマッピング型に格納し，
1文を形態素（マッピング型）のリストとして表現せよ．第4章の残りの問題では，ここで作ったプログラムを活用せよ．
'''
#表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音
with open('neko.txt.mecab') as f:
    #文章全体から文ごとに取り出したもののリスト
    morphemes_list = []
    #一文の形態素のリスト
    morphemes = []
    for line in f:
        line_processed = line.strip('\n')
        if line_processed == 'EOS':
            break
        else:
            columns = line_processed.split('\t')
            cols = columns[1].split(',')
            #形態素の辞書
            morpheme = {}
            #表層形
            morpheme['surface'] = columns[0]
            #原形
            morpheme['base'] = cols[6]
            #品詞
            morpheme['pos'] = cols[0]
            #品詞細分類1
            morpheme['pos1'] = cols[1]
            morphemes.append(morpheme)
            if morpheme['pos1'] == '句点':
                morphemes_list.append(morphemes)
                morphemes = []
            else : pass

    for morphemes in morphemes_list:
        print(morphemes)
