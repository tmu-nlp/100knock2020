#エンドポイントとはAPIにアクセスするためのURIのこと。例えば、QiitaのAPIで自分の情報を
#取得する時のエンドポイントは以下となる。
#http://qiita.com/api/v2/users/nagaokakenichi
#https://www.mediawiki.org/wiki/API:Imageinfo/ja
#https://hibiki-press.tech/learn_prog/python/requests_module/1882
'''
29. 国旗画像のURLを取得するPermalink
テンプレートの内容を利用し，国旗画像のURLを取得せよ．
（ヒント: MediaWiki APIのimageinfoを呼び出して，ファイル参照をURLに変換すればよい）
'''
import requests
import json
import re

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
    pattern = re.compile(r'\{\{仮リンク\|(.+?)\|(?:.+)\}\}')
    text = pattern.sub(r'\1', text)
    #Template:lang {{lang|言語タグ|文字列}}
    pattern = re.compile(r'\{\{lang\|.*\|(.+)\}\}')
    text = pattern.sub(r'\1', text)
    #Template:0 {{0}} ({{0}}との記述で、不可視な数字の0が挿入されます。表中で数字の
    #桁を揃える用途に使用できます。音声読み上げ式ユーザーエージェントでも読み上げられません。)
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

def get_image_url(image_name):
    URL = "http://ja.wikipedia.org/w/api.php"   #エンドポイント
    S = requests.Session()
    PARAMS = {
        #出力形式(JSON)
        "format" : "json",
        #query:特定のウィキとそこに収載されたデータからの情報を取得
        "action" : "query",
        #prop:プロパティとは、カテゴリのようなページに関するデータと、画像やリンクのようなページの内容のこと
        #imageinfo:ファイル情報と更新履歴を返す
        "prop"  : "imageinfo",
        #iiprop:どのファイル情報を取得するか
        #url:ファイルと説明ページへのURLを与える
        "iiprop" : "url",
        "titles" : "File:" + image_name
    }
    #リクエストは自動的に unicode にデコードされている.
    #json形式にデコード ===> 辞書型に変換
    imageinfo = S.get(url = URL, params = PARAMS).json()
    return imageinfo["query"]["pages"]["-1"]["imageinfo"][0]["url"]

def main():
    text = extract_uk()
    text_basic_info = extract_basic_info(text)
    text_processed = rm_markup(text_basic_info)
    dic = make_dict(text_processed)

    URL = get_image_url(dic['国旗画像'])
    print('URL:', URL)

if __name__ == "__main__":
    main()

'''
{
    "continue": {
        "iistart": "2019-09-10T16:52:58Z",
        "continue": "||"
    },
    "query": {
        "normalized": [
            {
                "from": "File:Flag of the United Kingdom.svg",
                "to": "\u30d5\u30a1\u30a4\u30eb:Flag of the United Kingdom.svg"
            }
        ],
        "pages": {
            "-1": {
                "ns": 6,
                "title": "\u30d5\u30a1\u30a4\u30eb:Flag of the United Kingdom.svg",
                "missing": "",
                "known": "",
                "imagerepository": "shared",
                "imageinfo": [
                    {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/a/ae/Flag_of_the_United_Kingdom.svg",
                        "descriptionurl": "https://commons.wikimedia.org/wiki/File:Flag_of_the_United_Kingdom.svg",
                        "descriptionshorturl": "https://commons.wikimedia.org/w/index.php?curid=347935"
                    }
                ]
            }
        }
    }
}
'''
