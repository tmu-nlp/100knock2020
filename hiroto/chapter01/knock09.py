#https://note.nkmk.me/python-random-shuffle/
import random

def typoglycemia(str):
    words = str.split()
    list = []
    for word in words:
        if len(word) <= 4:
            list.append(word)
        else:
            head = word[0]
            tail = word[len(word) - 1]
            temp = word[1 : len(word) - 1]
            temp_r = ''.join(random.sample(temp, len(temp)))
            word_r = head + temp_r + tail
            list.append(word_r)

    result = ' '.join(list)

    return result

str = "I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
print(typoglycemia(str))
