# 02. 「パトカー」＋「タクシー」＝「パタトクカシーー」
# 「パトカー」＋「タクシー」の文字を先頭から交互に連結して文字列「パタトクカシーー」を得よ．

if __name__ == "__main__":
    str1 = "パトカー"
    str2 = "タクシー"
    ans1 = ""
    for i in range(len(str1)):
        ans1 += str1[i] + str2[i]
    print(f"{ans1=}")

    ans2 = ""
    for s1, s2 in zip(str1, str2):
        ans2 += s1 + s2
    print(f"{ans2=}")

# 実行結果
# ans1='パタトクカシーー'
# ans2='パタトクカシーー'