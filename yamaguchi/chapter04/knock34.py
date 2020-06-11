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
def noun(nuko):
    result = []
    content = []

    for i in nuko:
        # 名詞の連接(連続して出現する名詞)を最長一致で抽出する
        # 名詞の場合，「content」に格納
        if i['pos'] == '名詞':
            content.append(i['surface'])
        # 名詞ではなく「content」の大きさが2以上のとき，
        # 「content」の内容を基に「result」に格納し，「content」を空にする．
        elif len(content) >= 2:
            result.append(''.join(content))
            content = []
        # 名詞ではなく「content」の大きさが1以下のときは何もしない
        else:
            content = []
    return result

def neko():
    with open(file, encoding='utf-8') as f:
        # 「split()」でデータを分割
        text = f.read().split('EOS\n')
    # 不要な行を「filter()」で除去
    text = list(filter(lambda x: x != '', text))
    text = [mecab(nuko) for nuko in text]
    answer = [noun(nuko) for nuko in text]

    print(answer)

if __name__ == '__main__':
    neko()
