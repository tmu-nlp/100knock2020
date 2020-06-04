# 乱数を生成するための「random」をインポート
import random

def shuffle(text):
    # 長さが4以下の単語はそのまま出力
    if len(text) <= 4:
        return text
    # 長さが4より大きいとき，各単語の先頭と末尾の文字以外をランダムに並び替える．
    else:
        first = text[0]
        last = text[-1]
        # random.sampleでは、重複することなくランダムに選択している
        others = random.sample(list(text[1:-1]), len(text[1:-1]))
        return "".join([first] + others + [last])

text = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."

result = [shuffle(count) for count in text.split()]

print(result)