#https://www.pytry3g.com/entry/N-gram
def n_gram(n, str):
    list_c, list_w = [], []
    #character
    index = 0
    for i in range(len(str) - n + 1):
        list_c.append(str[index : index + n])
        index += 1

    #word
    index = 0
    words = [word.strip(',.') for word in str.split()]
    for i in range(len(words) - n + 1):
        list_w.append(words[index : index + n])
        index += 1

    return list_c, list_w

str = 'I am an NLPer'

list_c, list_w = n_gram(1, str)
print('文字uni : ', list_c)
print('単語uni : ', list_w)

list_c, list_w = n_gram(2, str)
print('文字bi : ', list_c)
print('単語bi : ', list_w)

list_c, list_w = n_gram(3, str)
print('文字tri : ', list_c)
print('単語tri : ', list_w)
