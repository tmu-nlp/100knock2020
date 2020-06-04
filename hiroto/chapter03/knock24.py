'''
24. ファイル参照の抽出Permalink
記事から参照されているメディアファイルをすべて抜き出せ．
'''
import json
import re
#re.DOTALL : .（ワイルドカード）の対象に改行を加える。
#[[ファイル:Wikipedia-logo-v2-ja.png|thumb|説明文]]
def extract_uk():
    with open('jawiki-country.json') as file:
        for line in file:
            line_dict = json.loads(line)
            if line_dict['title'] == 'イギリス':
                text = line_dict['text']
            else: pass

    return text

def main():
    text = extract_uk()
    pattern = re.compile(r'''
        #普通のメディアファイル
        \[\[
        (?:ファイル)
        :
        (.+?)
        (?:\|.*)?
        \]\]
        ''', re.VERBOSE)
    list1 = pattern.findall(text)

    #世界遺産のところのテキストを抜き出す。
    pattern = re.compile(r'''
        ^
        <gallery>\n
        (.+?)
        </gallery>
        $
        ''', re.VERBOSE + re.DOTALL + re.MULTILINE)
    text_heritage = pattern.search(text).group(1)

    pattern = re.compile(r'''
        ^
        (.+?)
        \|
        .+
        $
        ''', re.VERBOSE + re.MULTILINE )
    list2 = pattern.findall(text_heritage)
    
    list = list1 + list2
    for f in list:
        print(f)

if __name__ == "__main__":
    main()

'''
===世界遺産===
イギリス国内には、[[国際連合教育科学文化機関|ユネスコ]]の[[世界遺産]]リストに登録された文化遺産が21件、自然遺産が5件ある。詳細は、[[イギリスの世界遺産]]を参照。
<gallery>
PalaceOfWestminsterAtNight.jpg|ウェストミンスター宮殿
Westminster Abbey - West Door.jpg|[[ウェストミンスター寺院]]
Edinburgh Cockburn St dsc06789.jpg|[[エディンバラ旧市街|エディンバラの旧市街]]・[[エディンバラ新市街|新市街]]
Canterbury Cathedral - Portal Nave Cross-spire.jpeg|[[カンタベリー大聖堂]]
Kew Gardens Palm House, London - July 2009.jpg|[[キューガーデン|キュー王立植物園]]
2005-06-27 - United Kingdom - England - London - Greenwich.jpg|[[グリニッジ|マリタイム・グリニッジ]]
Stonehenge2007 07 30.jpg|[[ストーンヘンジ]]
Yard2.jpg|[[ダラム城]]
Durham Kathedrale Nahaufnahme.jpg|[[ダラム大聖堂]]
Roman Baths in Bath Spa, England - July 2006.jpg|[[バース市街]]
Fountains Abbey view02 2005-08-27.jpg|[[ファウンテンズ修道院]]跡を含む[[スタッドリー王立公園]]
Blenheim Palace IMG 3673.JPG|[[ブレナム宮殿]]
Liverpool Pier Head by night.jpg|[[海商都市リヴァプール]]
Hadrian's Wall view near Greenhead.jpg|[[ローマ帝国の国境線]] ([[ハドリアヌスの長城]])
London Tower (1).JPG|[[ロンドン塔]]
</gallery>
'''
