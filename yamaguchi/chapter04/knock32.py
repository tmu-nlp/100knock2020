file = 'neko.txt.mecab'

def mecab(nuko):
    result = []

    for line in nuko.split('\n'):
        if line == '':
            return result
        (surface, tmp) = line.split('\t') # 水平タブ
        tmp = tmp.split(',')
        # 出力フォーマットは
        # 「表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,
        # 活用形,原形,読み,発音」の順
        d = {
            'surface': surface,
            'base': tmp[6],
            'pos': tmp[0],
            'pos1': tmp[1]
        }
        result.append(d)

# 「x['pos']=='動詞'」の「base」部分を取り出す
def verb(nuko):
    result = list(filter(lambda x: x['pos'] == '動詞', nuko))
    result = [r['base'] for r in result]
    return result

def neko():
    with open(file, encoding='utf-8') as f:
        # 「split()」でデータを分割
        text = f.read().split('EOS\n')
    # 不要な行を「filter()」で除去
    text = list(filter(lambda x: x != '', text))
    text = [mecab(nuko) for nuko in text]
    answer = [verb(nuko) for nuko in text]

    print(answer[5])

if __name__ == '__main__':
    neko()
