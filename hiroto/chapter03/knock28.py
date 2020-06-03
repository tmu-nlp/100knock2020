'''
28. MediaWikiマークアップの除去Permalink
27の処理に加えて，テンプレートの値からMediaWikiマークアップを可能な限り除去し，
国の基本情報を整形せよ．
'''
import json
import re

#内部リンク
#[[記事名]]
#[[記事名|表示文字]]
#[[記事名#節名|表示文字]]

def extract_uk():
    with open('jawiki-country.json') as file:
        for line in file:
            line_dict = json.loads(line)
            if line_dict['title'] == 'イギリス':
                text = line_dict['text']
            else: pass
    return text

def extract_basic_info(text):
    text_basic_info = re.search(r'''
        ^\{\{基礎情報\s国\n
        (.+?)
        ^\}\}$
        ''', text, re.VERBOSE + re.MULTILINE + re.DOTALL).group(1)

    return text_basic_info

def make_dict(text):
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

    li = pattern.findall(text)
    for line in li:
        dic[line[0]] = line[1]

    return dic

def rm_markup(text):
    #Template:仮リンク {{仮リンク|日本語版項目名|言語プレフィックス|他言語版項目名|...}}
    #('他元首等氏名2', '{{仮リンク|リンゼイ・ホイル|en|Lindsay Hoyle}}')
    pattern = re.compile(r'\{\{仮リンク\|(.+?)\|(?:.+)\}\}')
    text = pattern.sub(r'\1', text)
    #Template:lang {{lang|言語タグ|文字列}}
    #('標語', '{{lang|fr|[[Dieu et mon droit]]}}<br />（[[フランス語]]:[[Dieu et mon droit|神と我が権利]]）')
    pattern = re.compile(r'\{\{lang\|.*\|(.+)\}\}')
    text = pattern.sub(r'\1', text)
    #Template:0 {{0}} ({{0}}との記述で、不可視な数字の0が挿入されます。表中で数字の
    #桁を揃える用途に使用できます。音声読み上げ式ユーザーエージェントでも読み上げられません。)
    #('確立年月日2', '1707年{{0}}5月{{0}}1日')
    text = re.sub(r'\{\{0\}\}', r'', text)
    #メディアファイル抽出
    pattern = re.compile(r'''
        \[\[
        (?:ファイル)
        :
        (.+?)
        (?:\|.*)?
        \]\]
        ''', re.VERBOSE)
    text = pattern.sub(r'\1', text)
    #パイプ付きリンク, [[鉄道駅|駅]]とすると、駅と表示され「鉄道駅」にリンクされる
    pattern = re.compile(r'\[\[.*\|(.*)\]\]')
    text = pattern.sub(r'\1', text)

    pattern = re.compile(r"""
        '{2,5}  #強調
        |
        <(.*)ref(.*)>   #脚注
        |
        <(.*)br(.*)>    #段落
        |
        \[\[    #内部リンク
        |
        \]\]    #内部リンク
        """, re.VERBOSE)
    text = pattern.sub(r'', text)

    return text

def main():
    text = extract_uk()
    text_basic_info = extract_basic_info(text)
    text_processed = rm_markup(text_basic_info)
    dic = make_dict(text_processed)

    for tpl in dic.items():
        print(tpl)

if __name__ == "__main__":
    main()
