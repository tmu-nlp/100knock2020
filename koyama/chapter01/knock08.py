# 08. 暗号文
# 与えられた文字列の各文字を，以下の仕様で変換する関数cipherを実装せよ．
# 英小文字ならば(219 - 文字コード)の文字に置換
# その他の文字はそのまま出力
# この関数を用い，英語のメッセージを暗号化・復号化せよ．

def cipher(target):
    ans = ""
    for char in target:
        if char.islower():
            ans += chr(219 - ord(char))
        else:
            ans += char
    return ans

if __name__ == "__main__":
    str1 = "Colorless green ideas sleep furiously."
    ans1 = cipher(str1)
    ans2 = cipher(ans1)
    print(f"Cipher:{ans1}")
    print(f"Plaintext:{ans2}")

# 実行結果
# Cipher:Cloliovhh tivvm rwvzh hovvk ufirlfhob.
# Plaintext:Colorless green ideas sleep furiously.