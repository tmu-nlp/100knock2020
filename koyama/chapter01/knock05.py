# 05. n-gram
# 与えられたシーケンス（文字列やリストなど）からn-gramを作る関数を作成せよ．
# この関数を用い，”I am an NLPer”という文から単語bi-gram，文字bi-gramを得よ．

from collections import defaultdict

def n_gram(target, n=2, t="word"):
    ans = {}
    ans = defaultdict(int, ans)
    if t == "word":
        words = target.split()
        for i in range(len(words) - n + 1):
            ans[' '.join(words[i:i+n:])] += 1
        return ans
    elif t == "char":
        for i in range(len(target) -n + 1):
            ans[''.join(target[i:i+n])] += 1
        return ans

if __name__ == "__main__":
    str1 = "I am an NLPer"
    word_bi_gram = n_gram(str1)
    char_bi_gram = n_gram(str1, t="char")
    print(f"{word_bi_gram=}")
    print(f"{char_bi_gram=}")

# 実行結果
# word_bi_gram=defaultdict(<class 'int'>, {'I am': 1, 'am an': 1, 'an NLPer': 1})
# char_bi_gram=defaultdict(<class 'int'>, {'I ': 1, ' a': 2, 'am': 1, 'm ': 1, 'an': 1, 'n ': 1, ' N': 1, 'NL': 1, 'LP': 1, 'Pe': 1, 'er': 1})