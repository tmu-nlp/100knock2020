s = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. ' \
    'New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
indeces = [1, 5, 6, 7, 8, 9, 15, 16, 19]
indeces = [index - 1 for index in indeces]
words = s.split(' ')

res = {}
for (i, word) in enumerate(words):
    if i in indeces:
        res[word[:1]] = i + 1
        indeces.pop(0)
    else:
        res[word[:2]] = i + 1
print(res)
