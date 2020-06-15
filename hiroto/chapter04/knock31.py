'''
31. 動詞Permalink
動詞の表層形をすべて抽出せよ．
'''
#knock30でしたことと同じ
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

def extract_verb(morphemes):
    verbs = []
    for line in morphemes:
        for morpheme in line:
            if morpheme['pos'] == '動詞':
                verbs.append(morpheme['surface'])
            else: pass
    return verbs

def main():
    verbs = extract_verb(mapping())
    print(verbs)

if __name__ == "__main__":
    main()
