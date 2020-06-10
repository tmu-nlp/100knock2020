'''
33. 「AのB」Permalink
2つの名詞が「の」で連結されている名詞句を抽出せよ．
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

#名詞句を抽出する
def extract_NP(morphemes_list):
    NPs = []
    for morphemes in morphemes_list:
        for i in range(len(morphemes)):
            if morphemes[i]['base'] == 'の'\
                and morphemes[i]['pos'] == '助詞'\
                and morphemes[i-1]['pos'] == '名詞'\
                and morphemes[i+1]['pos'] == '名詞':
                NP = morphemes[i-1]['surface'] + morphemes[i]['surface']\
                    + morphemes[i+1]['surface']
                NPs.append(NP)
            else: pass
    return NPs


def main():
    NPs = extract_NP(mapping())
    print(NPs)
    print(len(NPs))

if __name__ == "__main__":
    main()
