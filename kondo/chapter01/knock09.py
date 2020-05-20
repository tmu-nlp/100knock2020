import random

def typoglycemia(str):
    result = []
    for word in str:
        if len(word) > 4:
            word_list = list(word)
            shuf_word_list = word_list[1:-1]
            random.shuffle(shuf_word_list)
            word_list[1:-1] = shuf_word_list
            result.append("".join(word_list))
        else:
            result.append(word)

    return " ".join(result)

str = input().split(" ")

print(typoglycemia(str))
