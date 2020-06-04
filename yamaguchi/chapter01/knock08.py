def cipher(text):
    # 英小文字なら(219-文字コード)の文字に置換，その他の文字はそのまま出力．
    # 文字コードを得るために「ord()」を用いる．
    # 文字コードから文字を得るために「chr()」を用いる．
    text = [chr(219 - ord(w)) if 97 <= ord(w) <= 122 else w for w in text]
    return "".join(text)

text = "Tokyo Metropolitan University"

result = cipher(text)

print("入力された文字列: " + text)
print("変換後の文字列: " + result)