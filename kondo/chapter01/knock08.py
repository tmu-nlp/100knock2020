def cipher(str):
    ans = ""
    for charact in str:
        if charact.islower():
            #chr アスキーコードから文字
            #ord 文字からアスキーコード
            ans += chr(219 - ord(charact))
        else:
            ans += charact

    return ans

str = input()
crypt = cipher(str)
decrypt = cipher(crypt)
print(crypt)
print(decrypt)
