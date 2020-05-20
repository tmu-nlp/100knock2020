def n_gram(n, str):
    lens = len(str)
    list = []
    for i in range(lens - n + 1):
        list.append(str[i:i + n])

    return list

str1 = "I am an NLPer"
str2 = str1.split(" ")

#単語bi-gram
word_bi_gram = n_gram(2, str2)

#文字bi-gram
chracter_bi_gram = n_gram(2, str1)

print("単語bi-gram: {}\n文字bi-gram: {}".format(word_bi_gram, chracter_bi_gram))
