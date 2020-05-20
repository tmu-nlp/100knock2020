#https://note.nkmk.me/python-capitalize-lower-upper-title/
#ord(): アスキーコードを取得
#chr(): アスキーコードから文字へ
def cipher(str):
    result = ''
    for i in range(len(str)):
        if str[i].islower():
            result += chr((219 - ord(str[i]))) # 219 - x = y -> x = 219 - y
        else:
            result += str[i]

    return result

str = 'Now I need a drink, alcoholic of course'
print('元の文 : ' + str)
txt = cipher(str)
print('暗号文 : ' + txt)
print('復号文 : ' + cipher(txt))
