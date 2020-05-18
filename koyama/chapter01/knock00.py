# 00. 文字列の逆順
# 文字列"stressed"の文字を逆に（末尾から先頭に向かって）並べた文字列を得よ．

if __name__ == "__main__":
    str1 = "stressed"
    ans = str1[::-1]
    print(f"{ans=}")

# 実行結果
# ans='desserts'
