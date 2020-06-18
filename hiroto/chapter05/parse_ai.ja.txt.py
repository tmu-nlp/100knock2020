# coding: utf_8 ai.ja.txt.parsed
import re, io
import CaboCha
format = CaboCha.FORMAT_LATTICE
with open('ai.ja.txt') as r_file\
    , open('ai.ja.txt.parsed', mode = 'w') as w_file\
    , open('processed_ai.ja.txt', mode='w') as newfile:
    cabocha = CaboCha.Parser("-d /var/lib/mecab/dic/ipadic-utf8")
    text = r_file.read()

    c = re.compile(r'\n(?!\n)')
    text = c.sub(r'', text)

    text = list(text)

    #大文字
    upper = {chr(0xFF21 + ch):chr(0x0041 + ch) for ch in range(26)}
    #小文字
    lower = {chr(0xFF41 + ch):chr(0x0061 + ch) for ch in range(26)}
    #数字
    number = {chr(0xFF10 + ch):chr(0x0030 + ch) for ch in range(10)}
    #全角から半角への変換テーブル
    zen2han = str.maketrans({**upper, **lower, **number})
    #左括弧に対応する右括弧
    dic = {'（':'）', '(':')', '「':'」'}
    flag = False
    right = -1
    cnt = 0
    for c in text:
        c = c.translate(zen2han)
        #cが左括弧なら
        if c in dic.keys():
            #右括弧
            right = dic[c]
            flag = True
        if c == '。':
            #括弧内の'。'は改行しない
            if flag == True: pass
            else:
                text[cnt] = '。\n'
        if c == right: flag = False
        cnt += 1

    text = ''.join(text)

    #段落のための改行を消す
    c = re.compile(r'(\n)+')
    text = c.sub(r'\n', text)
    newfile.write(text)

    f = io.StringIO(text)
    for line in f:
        parsed_line = cabocha.parse(line)
        w_file.write(parsed_line.toString(format))
