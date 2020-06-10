'''
32. 動詞の原形Permalink
動詞の原形をすべて抽出せよ．
'''
def mapping():
    with open('neko.txt.mecab') as file:
        morphemes_list = []
        morphemes = []
        for line in file:
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
                verbs.append(morpheme['base'])
            else: pass
    return verbs


def main():
    verbs = extract_verb(mapping())
    print(verbs)

if __name__ == "__main__":
    main()
