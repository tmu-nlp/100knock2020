# 01. 「パタトクカシーー」Permalink
# 「パタトクカシーー」という文字列の1,3,5,7文字目を取り出して連結した文字列を得よ．

if __name__ == "__main__":
    str1 = "パタトクカシーー"
    ans1 = str1[0::2]
    ans2 = str1[1::2]
    print(f"{ans1=}")
    print(f"{ans2=}")

# 実行結果
# ans1='パトカー'
# ans2='タクシー'