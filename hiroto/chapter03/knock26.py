'''
26. 強調マークアップの除去Permalink
25の処理時に，テンプレートの値からMediaWikiの強調マークアップ（弱い強調，強調，強い強調のすべて）
を除去してテキストに変換せよ（参考: マークアップ早見表）．
'''
import json
import re
#アポストロフィー2個ずつで囲むと斜体になります。

#アポストロフィー3個で太字になります。

#アポストロフィー5個で斜体かつ太字になります。

def extract_uk():
    with open('jawiki-country.json') as file:
        for line in file:
            line_dic = json.loads(line)
            if line_dic['title'] == 'イギリス':
                text = line_dic['text']
            else: pass

    return text

def extract_basic_info(text):
    text_basic_info = re.search(r'''
        ^\{\{基礎情報\s国\n
        (.+?)
        ^\}\}$
        ''', text, re.VERBOSE + re.MULTILINE + re.DOTALL).group(1)

    dic = {}
    pattern = re.compile(r'''
        ^\|
        (.+?)
        (?:\s*)
        =
        (?:\s*)
        (.+?)
        (?:         # キャプチャ対象外のグループ開始
        (?=\n\|)    # 改行+'|'の手前（肯定の先読み）
        |
        (?=\n$)   # または、改行+終端の手前（肯定の先読み）
        )
        ''', re.MULTILINE + re.VERBOSE + re.DOTALL)

    lis = pattern.findall(text_basic_info)
    for line in lis:
        dic[line[0]] = line[1]

    return dic

def main():
    text = extract_uk()
    #アポストロフィーを消す
    text_processed = re.sub(r"'{2,5}", r"", text)
    dic = extract_basic_info(text_processed)

    for tpl in dic.items():
        print(tpl)

if __name__ == "__main__":
    main()

"""
25
('確立形態4', "現在の国号「'''グレートブリテン及び北アイルランド連合王国'''」に変更")
26
('確立形態4', '現在の国号「グレートブリテン及び北アイルランド連合王国」に変更')
"""