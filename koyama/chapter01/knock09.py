# 09. Typoglycemia
# スペースで区切られた単語列に対して，各単語の先頭と末尾の文字は残し，それ以外の文字の順序をランダムに並び替えるプログラムを作成せよ．
# ただし，長さが４以下の単語は並び替えないこととする．
# 適当な英語の文（例えば”I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind .”）を与え，その実行結果を確認せよ．

import random

if __name__ == "__main__":
    str1 = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
    words = str1.split(" ")
    ans = []
    for word in words:
        if len(word) <= 4:
            ans.append(word)
        else:
            chars = list(word)
            first = chars[0]
            last = chars[-1] 
            chars = chars[1:len(chars) - 1]
            random.seed(0)
            random.shuffle(chars)
            ans.append(first + ''.join(chars) + last)
    ans = ' '.join(ans)
    print(f"{ans=}")

# 実行結果
# ans='I cnluo’dt bilevee that I colud alutclay usdtenrnad what I was rdaenig : the pmeenhoanl poewr of the huamn mind .'