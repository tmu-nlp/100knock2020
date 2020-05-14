# 03. 円周率
# “Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.”という文を単語に分解し，各単語の（アルファベットの）文字数を先頭から出現順に並べたリストを作成せよ．

import re

if __name__ == "__main__":
    str1 = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
    words = str1.split(" ")
    ans = []
    for word in words:
        word = re.sub("[^a-zA-Z]", "", word) # アルファベット以外を削除する
        ans.append(len(word))
    print(f"{ans=}")

# 実行結果
# ans=[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9]